import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

def list_atlas_files(atlas_dir: str) -> List[Path]:
    paths = []
    for p in sorted(Path(atlas_dir).iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith('atlas_seg_'):
            continue
        if p.suffix.lower() in {'.nii'}:
            paths.append(p)
        elif p.suffix.lower() == '.gz' and p.name.endswith('.nii.gz'):
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f'No atlas files found in {atlas_dir}. Expected atlas_n*.nii/.nii.gz files.')
    print(f'Found {len(paths)} atlas volumes in {atlas_dir}', flush=True)
    return paths


def load_atlas(path: Path) -> Tuple[torch.Tensor, np.ndarray, nib.Nifti1Header]:
    print(f'Loading atlas: {path}', flush=True)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(data)
    affine = img.affine
    header = img.header.copy()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        pass
    elif tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    else:
        raise ValueError(f'Unexpected tensor shape for {path}: {tuple(tensor.shape)}')

    if tensor.shape[0] != 1:
        raise ValueError(f'Expected C=1 for {path}, got shape {tuple(tensor.shape)}')

    return tensor, affine, header

"""
def zscore_volume(x: torch.Tensor) -> torch.Tensor:
    # per-volume z-score normalization.
    mean = x.mean()
    std = x.std()
    std = torch.clamp(std, min=1e-6)
    return (x - mean) / std
"""

def normalize_with_stats(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.clamp(std, min=1e-6)
    return (x - mean) / std


def denormalize_with_stats(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


class AtlasBankDataset(torch.utils.data.Dataset):
    def __init__(self, atlas_dir: str):
        self.paths = list_atlas_files(atlas_dir)
        # keep affine/header of reference atlas for writing samples
        self.ref_tensor, self.ref_affine, self.ref_header = load_atlas(self.paths[0])
        self.ref_mask = (self.ref_tensor != 0) & torch.isfinite(self.ref_tensor)
        # compute atlas bank mean/std for invertible normalization
        self.global_mean, self.global_std = self._compute_global_stats()

    def _compute_global_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        print('Computing global mean/std over atlas bank...', flush=True)
        total_sum = 0.0
        total_sumsq = 0.0
        total_count = 0

        for p in self.paths:
            tensor, _, _ = load_atlas(p)
            data = tensor.double()

            mask = (data != 0) & torch.isfinite(data)
            vals = data[mask]

            if vals.numel() == 0:
                continue

            total_sum += vals.sum().item()
            total_sumsq += (vals * vals).sum().item()
            total_count += vals.numel()

        mean = total_sum / max(total_count, 1)
        var = (total_sumsq / max(total_count, 1)) - (mean * mean)
        std = float(np.sqrt(max(var, 1e-12)))
        print(f'Global mean/std: {mean:.6g}, {std:.6g}', flush=True)
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # load and normalize one volume with global stats
        tensor, _, _ = load_atlas(self.paths[idx])

        mask = (tensor != 0) & torch.isfinite(tensor)
        tensor_norm = torch.zeros_like(tensor)
        tensor_norm[mask] = normalize_with_stats(tensor[mask], self.global_mean, self.global_std)
        return tensor_norm


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 16, out_ch: int = 1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

def flow_target(z: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, beta_eps: float) -> torch.Tensor:
    # flow target u*(x_t | z, t) with beta clamped for stability.
    alpha = t
    beta = 1.0 - t
    beta_safe = torch.clamp(beta, min=beta_eps)
    alpha_dot = 1.0
    beta_dot = -1.0

    coef_z = alpha_dot - (beta_dot / beta_safe) * alpha
    coef_x = beta_dot / beta_safe
    return coef_z * z + coef_x * x_t


def train(atlas_dir: str, out_dir: str, epochs: int, batch_size: int, lr: float,
          beta_eps: float, save_every: int, seed: int) -> Tuple[nn.Module, AtlasBankDataset]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('Initializing dataset and dataloader...', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    dataset = AtlasBankDataset(atlas_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print('Initializing model and optimizer...', flush=True)
    model = UNet3D(in_ch=2, base_ch=16, out_ch=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    log_path = os.path.join(out_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'loss'])

    global_step = 0
    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}', flush=True)
        for batch_idx, batch in enumerate(loader, start=1):
            print(f'Epoch {epoch} batch {batch_idx}/{len(loader)}', flush=True)
            # sample atlas z ~ p_atlas and noise eps ~ N(0,1).
            z = batch
            t = torch.rand((z.shape[0], 1, 1, 1, 1))
            eps = torch.randn_like(z)
            # construct x_t = alpha*z + beta*eps with alpha=t, beta=1-t.
            x_t = t * z + (1.0 - t) * eps

            # compute the target vector field u*.
            target = flow_target(z, x_t, t, beta_eps)

            # predict u_hat = u_theta(t, x_t) (time channel concatenated).
            t_chan = t.expand(x_t.shape)
            u_hat = model(torch.cat([x_t, t_chan], dim=1))
            mask = (z != 0) & torch.isfinite(z)
            loss = F.mse_loss(u_hat[mask], target[mask])

            # clear old gradient from last step
            optimizer.zero_grad()
            # derivate of loss w.r.t. model parameters
            loss.backward()
            # optimzer takes step based on the current gradient
            optimizer.step()

            global_step += 1
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, float(loss.detach().cpu().item())])

        if epoch % save_every == 0:
            print(f'Saving checkpoint at epoch {epoch}', flush=True)
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(out_dir, 'optimizer.pt'))

    return model, dataset


def sample_after_train(model: nn.Module, dataset: AtlasBankDataset, out_dir: str, ode_steps: int):
    model.eval()
    print('Sampling from trained flow...', flush=True)
    # start from Gaussian noise and integrate dX/dt = u_theta(t, X) with Runge-Kutta.
    ref = dataset[0]
    x = torch.randn_like(ref).unsqueeze(0)

    dt = 1.0 / ode_steps
    with torch.no_grad():
        for i in range(ode_steps):
            t0 = torch.tensor((i / ode_steps), dtype=torch.float32).view(1, 1, 1, 1, 1)
            t0_chan = t0.expand(x.shape)

            # first slope 
            u1 = model(torch.cat([x, t0_chan], dim=1))

            # midpoint state
            x_mid = x + 0.5 * dt * u1

            # time at midpoint 
            t_mid = torch.tensor(((i + 0.5) / ode_steps), dtype=torch.float32).view(1, 1, 1, 1, 1)
            t_mid_chan = t_mid.expand(x.shape)

            # second slope
            u2 = model(torch.cat([x_mid, t_mid_chan], dim=1))

            # update
            x = x + dt * u2

    x1_norm = x.squeeze(0).detach().cpu()

    # denormalize sample so we can visualize
    mask = dataset.ref_mask.cpu()
    x1 = torch.zeros_like(x1_norm)
    x1[mask] = denormalize_with_stats(x1_norm[mask], dataset.global_mean, dataset.global_std)

    # save sample
    nii_path = os.path.join(out_dir, 'sample_x1.nii.gz')
    img = nib.Nifti1Image(x1.squeeze(0).numpy(), dataset.ref_affine, header=dataset.ref_header)
    nib.save(img, nii_path)

    # save normalized sample
    norm_path = os.path.join(out_dir, 'sample_x1_norm.nii.gz')
    img_norm = nib.Nifti1Image(x1_norm.squeeze(0).numpy(), dataset.ref_affine, header=dataset.ref_header)
    nib.save(img_norm, norm_path)


def main():
    parser = argparse.ArgumentParser(description='Flow Matching over atlas volumes')
    parser.add_argument('--atlas_dir', required=True, type=str, help='Atlas bank directory with .nii/.nii.gz files')
    parser.add_argument('--out_dir', required=True, type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta_eps', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--sample_after_train', action='store_true')
    parser.add_argument('--sample_only', action='store_true', help='Only run sampling using a saved model in out_dir.')
    parser.add_argument('--ode_steps', type=int, default=50)
    args = parser.parse_args()

    print('Starting Flow Matching training...', flush=True)
    if args.sample_only:
        dataset = AtlasBankDataset(args.atlas_dir)
        model = UNet3D(in_ch=2, base_ch=16, out_ch=1)
        model_path = os.path.join(args.out_dir, 'model.pt')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model checkpoint not found: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model, dataset = train(
            atlas_dir=args.atlas_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_eps=args.beta_eps,
            save_every=args.save_every,
            seed=args.seed,
        )

    if args.sample_after_train or args.sample_only:
        sample_after_train(model, dataset, args.out_dir, args.ode_steps)


if __name__ == '__main__':
    main()
