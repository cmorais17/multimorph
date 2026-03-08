import os
import sys
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import models
import pandas as pd
from dataloader import SubGroupLoader3D, PadtoDivisible
import layers
from typing import Tuple, List
import nibabel as nib
import itertools
import csv
import glob
import random

def warp_segmentation(segmentations: torch.Tensor,
                        warps: torch.Tensor,
                        interpolation_mode='nearest'
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Warps segmentations to the atlas space, and aggregates them.
    Args:
        segmentations: (B, G, C, H, W) torch.Tensor, segmentations to warp. One-hot representation.
        warps: (B, G, 3, H, W) torch.Tensor, warps to apply to the segmentations (3D)
        indices_to_warp: List[int], indices of segmentations to warp
        interpolation_mode: 'bilinear' or 'nearest'. Only use bilinear if segmentation is in one-hot format.
    Returns:
        warped_segmentation: (B, G, C, H, W) torch.Tensor, warped segmentations
        atlas_segmentation: (B, 1, 1, H, W) torch.Tensor, aggregated segmentations in the atlas
    '''
    if len(segmentations.shape) == 5:
        img_size = segmentations.shape[-3:]
        segmentations = segmentations.unsqueeze(0)
        warps = warps.unsqueeze(0)
    else:
        img_size = segmentations.shape[-3:]
        
    warp_layer = layers.group.Warp3d(img_size, mode=interpolation_mode)
    
    if interpolation_mode == 'nearest':
        # Check if segmentations are in one-hot. If so, convert to per-class
        if segmentations.shape[2] > 1:
            segmentations = torch.argmax(segmentations, dim=2, keepdim=True).to(torch.float)
        # warp the segmentations
        warped_segmentation = warp_layer(segmentations, warps)
        atlas_segmentation= torch.mode(warped_segmentation, dim=1, keepdim=True)[0]
    else:
        # make sure the segmentations are in one-hot format
        if segmentations.shape[2] == 1:
            raise ValueError("Segmentations should be in one-hot format for bilinear interpolation.")
            
        warped_segmentation = warp_layer(segmentations, warps)
        atlas_segmentation = torch.argmax(torch.mean(warped_segmentation, dim=1, keepdims=True), dim=2, keepdim=True)
        warped_segmentation = torch.argmax(warped_segmentation,dim=2,keepdim=True)

    return warped_segmentation, atlas_segmentation
    
def load_model(model_weights_path:str, img_size:list[int]):
    '''
    Load the 3D MultiMorph model and weights. Currently only supports CPU, and a fixed instantiation of the model.
    Args:
        model_weights_path: path to the model weights file
        img_size: size of the input images, should be a list of 3 integers [depth, height, width]
    Returns:
        mmnet: the MultiMorph model with the loaded weights
    Raises:
        FileNotFoundError: if the model weights file does not exist
    '''
    mmnet = models.GroupNet3D(in_channels=1, out_channels=3,img_size=img_size,
                              features=[32,128,128,128],do_mean_conv=True,diffeo_steps=5,do_half_res=True,
                              subtract_mean=True, do_instancenorm=True,summary_stat='mean',
                              checkpoint_model=False)
    
    # load the model weights
    if os.path.isfile(model_weights_path):
        print(f"Loading model weights from {model_weights_path}")
        checkpoint = torch.load(model_weights_path, map_location='cpu')
        mmnet.load_state_dict(checkpoint['state_dict'])
        
        return mmnet
    else:
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")
        
def build_atlas(model:torch.nn.Module, dataset:Dataset,
                device:torch.device
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Builds an atlas by warping the images in the dataset using the multimorph model.
    Args:
        model: the MultiMorph model to use for warping
        dataset: the dataset containing the images to warp
        device: the device to run the model on (CPU or GPU)
    Returns:
        atlas: the atlas image, averaged over the warped images
        atlas_segmentation: the atlas segmentation, averaged over the warped segmentations (if segmentations are provided)
    
    '''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # get the image size
    img_size = dataset._get_img_size()
    
    # create the 3D warp layer to warp images
    warp_layer = layers.group.Warp3d(img_size)
    
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            image = sample['image'].to(device)
            num_imgs = image.shape[1]
            
            segmentation = sample['segmentation']
            if segmentation is not None:
                segmentation = segmentation.to(device)
            else:
                segmentation = None
            
            
            start_time = time.time()
            # forward pass through the model. Get the warp field and warp the images.
            predicted_warp = model(image)
            warped_group = warp_layer(image, predicted_warp)
            
            # compute the atlas
            atlas = torch.mean(warped_group, dim=1, keepdim=True)
            
            # report time taken
            end_time = time.time()
            print(f"Processed {num_imgs} images in {end_time - start_time:.2f} seconds")
            
            # warp the segmentations
            if segmentation is not None:
                warped_segmentation, atlas_segmentation = warp_segmentation(
                    segmentations=segmentation,
                    warps=predicted_warp,
                    interpolation_mode='nearest'  # or 'bilinear' if segmentations are in one-hot format
                )
            else:
                atlas_segmentation = None
            
            return atlas, atlas_segmentation


def gather_oasis2d_slices(oasis_root: str) -> List[str]:
    pattern = os.path.join(oasis_root, '*', 'slice_norm.nii.gz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No slice_norm.nii.gz files found under {oasis_root}')
    return paths


def load_slice_2d(path: str) -> torch.Tensor:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f'Unexpected slice shape for {path}: {tuple(tensor.shape)}')
    return tensor


def build_atlas_2d(model: torch.nn.Module, image_paths: List[str], device: torch.device) -> torch.Tensor:
    # Load 2D slices into a [1, G, 1, H, W] tensor
    images = [load_slice_2d(p) for p in image_paths]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [B, G, C, H, W]
    img_size = list(images.shape[-2:])
    warp_layer = layers.group.Warp2d(img_size)

    model.eval()
    with torch.no_grad():
        predicted_warp = model(images)
        warped_group = warp_layer(images, predicted_warp)
        atlas = torch.mean(warped_group, dim=1, keepdim=True)
    return atlas.cpu()

def wrapper_build_atlas_bank(model_path, atlas_save_path, csv_path, img_header_name, segmentation_header_name, group_sizes: List[int]):
    '''
    Main wrapper function to build an atlas bank by inference on a pre-trained model.
    This function loads the model and weights, loads the dataset from a CSV file,
        and builds atlases for all combinations of the requested group sizes.
    Args:
        model_path: path to the pre-trained model weights
        atlas_save_path: path to save the atlas bank
        csv_path: path to the CSV file containing the list of images and segmentations
        img_header_name: header name for the image column in the CSV file
        segmentation_header_name: header name for the segmentation column in the CSV file (optional)
        group_sizes: list of group sizes (e.g., [2,3,4])
    Returns:
        None, saves the atlas bank to the specified path.
    '''
    
    # get the device. Currently on supports CPU
    device = torch.device('cpu')
    
    # load the CSV file with image paths
    csv_data = pd.read_csv(csv_path)
    # get the segmentation paths (if they exist)
    segmentations = csv_data[segmentation_header_name].tolist() if segmentation_header_name is not None else None
    images = csv_data[img_header_name].tolist()

    if len(images) < 2:
        raise ValueError('Need at least 2 images to build an atlas bank of pairs and triples.')

    group_sizes = sorted(set(group_sizes))
    if not group_sizes:
        raise ValueError('No group sizes specified. Use --group_sizes, e.g. "2,3,4".')
    if any(g <= 0 for g in group_sizes):
        raise ValueError(f"Invalid group sizes {group_sizes}. All sizes must be positive integers.")
    max_size = max(group_sizes)
    if max_size > len(images):
        raise ValueError(f"Requested group size {max_size}, but only {len(images)} images are available.")

    # load a dataset to get the image size
    first_group = list(range(group_sizes[0]))
    first_imgs = [images[i] for i in first_group]
    first_segs = [segmentations[i] for i in first_group] if segmentations is not None else None
    dataset = SubGroupLoader3D(data=first_imgs, labels=None,
                                   segmentations=first_segs, file_names=None, segmentation_to_one_hot=False,
                                   transform=PadtoDivisible())
    img_size = dataset._get_img_size()
    print(f"Image size: {img_size}")
    
    # get the model
    mmnet = load_model(model_path, img_size)
    mmnet = mmnet.to(device)

    os.makedirs(atlas_save_path, exist_ok=True)
    index_path = os.path.join(atlas_save_path, 'atlas_bank.csv')

    with open(index_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'group_size',
            'group_indices',
            'image_paths',
            'segmentation_paths',
            'atlas_path',
            'atlas_segmentation_path',
        ])

        for k in group_sizes:
            for combo in itertools.combinations(range(len(images)), k):
                group_imgs = [images[i] for i in combo]
                group_segs = [segmentations[i] for i in combo] if segmentations is not None else None

                dataset = SubGroupLoader3D(data=group_imgs, labels=None,
                                               segmentations=group_segs, file_names=None, segmentation_to_one_hot=False,
                                               transform=PadtoDivisible())
                group_img_size = dataset._get_img_size()
                if list(group_img_size) != list(img_size):
                    raise ValueError(
                        f"Image size mismatch for group {combo}. Expected {img_size}, got {group_img_size}."
                    )

                # build the atlas
                atlas, atlas_segmentation = build_atlas(mmnet, dataset, device)

                group_tag = '__'.join([f"i{idx}" for idx in combo])
                atlas_path = os.path.join(atlas_save_path, f"atlas_n{k}__{group_tag}.nii.gz")
                nib.save(nib.Nifti1Image(atlas.squeeze().numpy(), np.eye(4)), atlas_path)

                atlas_seg_path = ''
                if atlas_segmentation is not None:
                    atlas_seg_path = os.path.join(atlas_save_path, f"atlas_seg_n{k}__{group_tag}.nii.gz")
                    nib.save(nib.Nifti1Image(atlas_segmentation.squeeze().numpy(), np.eye(4)), atlas_seg_path)

                writer.writerow([
                    k,
                    ';'.join([str(i) for i in combo]),
                    ';'.join(group_imgs),
                    ';'.join(group_segs) if group_segs is not None else '',
                    atlas_path,
                    atlas_seg_path,
                ])

                print(f"Saved atlas for group indices {combo}")

    print(f"Atlas bank index saved to {index_path}")


def wrapper_build_atlas_bank_2d(model_path: str,
                                atlas_save_path: str,
                                oasis_root: str,
                                group_size: int = 30,
                                num_groups: int = 750,
                                seed: int = 0):
    """
    Build a 2D atlas bank from slice_norm.nii.gz files.
    Uses fixed group size and random grouping (with replacement across groups).
    """
    device = torch.device('cpu')
    os.makedirs(atlas_save_path, exist_ok=True)

    slice_paths = gather_oasis2d_slices(oasis_root)
    if group_size > len(slice_paths):
        raise ValueError(f'group_size={group_size} exceeds number of slices ({len(slice_paths)})')

    # Load one slice to get H,W
    ref = load_slice_2d(slice_paths[0])
    img_size = list(ref.shape[-2:])
    print(f'2D image size: {img_size}', flush=True)

    # Build 2D GroupNet
    mmnet = models.GroupNet(in_channels=1, out_channels=2, img_size=img_size,
                            features=[32, 64, 64, 64], do_mean_conv=True)
    if os.path.isfile(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        mmnet.load_state_dict(checkpoint['state_dict'])
    else:
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    mmnet = mmnet.to(device)

    rng = random.Random(seed)
    index_path = os.path.join(atlas_save_path, 'atlas_bank.csv')
    with open(index_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'group_size',
            'group_indices',
            'image_paths',
            'segmentation_paths',
            'atlas_path',
            'atlas_segmentation_path',
        ])

        for gi in range(num_groups):
            group_indices = rng.sample(range(len(slice_paths)), group_size)
            group_paths = [slice_paths[i] for i in group_indices]

            atlas = build_atlas_2d(mmnet, group_paths, device)
            atlas_path = os.path.join(atlas_save_path, f"atlas2d_g{gi:04d}.nii.gz")
            nib.save(nib.Nifti1Image(atlas.squeeze().numpy(), np.eye(4)), atlas_path)

            writer.writerow([
                group_size,
                ';'.join(str(i) for i in group_indices),
                ';'.join(group_paths),
                '',
                atlas_path,
                '',
            ])

            if (gi + 1) % 25 == 0:
                print(f'Saved {gi + 1}/{num_groups} 2D atlases', flush=True)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Build an atlas bank by inference on a pre-trained model')
    parser.add_argument('--model_path', type=str, default='./models/model_cvpr.pt', help='Path to the pre-trained model')
    parser.add_argument('--atlas_save_path', default='results/atlas_bank', type=str, help='Path to save the atlas bank')
    parser.add_argument('--csv_path', default='data/oasis_3d_data/metadata.csv', type=str, help='Path to the CSV file containing the list of images')
    parser.add_argument('--img_header_name', type=str, default='img_path', help='Header name for the image column in the CSV file')
    parser.add_argument('--segmentation_header_name', default=None, help='Header name for the segmentation column in the CSV file. \
                            Use None if no segmentations are provided.')
    parser.add_argument('--group_sizes', default='2,3,4', help='Comma-separated group sizes, e.g. "2,3,4"')
    parser.add_argument('--oasis2d', action='store_true', help='Build 2D atlas bank from data/oasisdata slice_norm.nii.gz')
    parser.add_argument('--oasis2d_root', default='data/oasisdata', help='Root folder for OASIS 2D slices')
    args = parser.parse_args()
    
    model_path = args.model_path
    atlas_save_path = args.atlas_save_path
    csv_path = args.csv_path
    img_header_name = args.img_header_name
    segmentation_header_name = args.segmentation_header_name
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(',') if x.strip()]
    
    if args.oasis2d:
        wrapper_build_atlas_bank_2d(model_path, atlas_save_path, args.oasis2d_root,
                                    group_size=30, num_groups=750, seed=0)
    else:
        wrapper_build_atlas_bank(model_path, atlas_save_path, csv_path, img_header_name, segmentation_header_name, group_sizes)
