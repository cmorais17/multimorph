#!/usr/bin/env python
# coding: utf-8

# # MultiMorph Demo on OASIS-1 (2D)
# Be sure to select the GPU in CoLab for training

# In[ ]:


# Repo already cloned; ensure you're running from the repo root
import os
import sys
from pathlib import Path

repo_root = Path('.').resolve()
src_dir = repo_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print('CWD:', os.getcwd())
print('Using src_dir:', src_dir)


# ### Load Libraries

# In[ ]:


# Assumes required deps are already installed: neurite, monai, nibabel, torch


# In[ ]:


# imports
import pathlib
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import trange, tqdm

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import neurite as ne
import models as models
import layers as layers
import losses as losses
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # OASIS-1 Data
# download OASIS-1 2D Data

# In[ ]:


# Using local OASIS-1 2D data from data/oasisdata
# See data/oasisdata/README.md if you need to download it separately.


# ### Load the OASIS Data
# stack the data together to a tensor to create a dataloader from. Specify the path where the oasis data lives

# In[ ]:


# specify the full path to the OASIS data
# expect data/oasisdata/OASIS_*/slice_norm.nii.gz

oasis_data_path = 'data/oasisdata'

if not pathlib.Path(oasis_data_path).exists():
    raise FileNotFoundError(f"Missing {oasis_data_path}. See data/oasisdata/README.md for download instructions.")

data_dirs = [d for d in pathlib.Path(oasis_data_path).iterdir() if d.is_dir()]
files = [d/'slice_norm.nii.gz' for d in data_dirs]

slices = []
for f in files:
    arr = nib.load(f).get_fdata()
    if arr.ndim == 3:
        arr = arr[..., 0]
    slices.append(torch.from_numpy(arr))

oasis_data = torch.stack(slices, dim=0).float().to(device)  # put all data on device

# get the segmentations (4-class)
seg_files = [d/'slice_seg4.nii.gz' for d in data_dirs]
seg_slices = []
for f in seg_files:
    arr = nib.load(f).get_fdata()
    if arr.ndim == 3:
        arr = arr[..., 0]
    seg_slices.append(torch.from_numpy(arr))

oasis_data_segmentation = torch.stack(seg_slices, dim=0).float().to(device)

# match expected orientation
# images: (N,H,W) -> (N,W,H) to match downstream assumptions
# segmentations: (N,H,W) -> (N,W,H), then add channel dim

oasis_data = oasis_data.transpose(2, 1)
oasis_data_segmentation = oasis_data_segmentation.transpose(2, 1).unsqueeze(1)

print(oasis_data.shape)
print(oasis_data_segmentation.shape)

oasis_img_size = list(map(int, list(oasis_data.shape[1:])))
print(oasis_img_size)


# # Training
# This notebook only trains and saves 2D model weights.
# 

# In[ ]:


from dataloader import GroupDataLoader, SubGroupLoader
from torch.utils.data import DataLoader


# number of images to sample at each training iteration
n_inputs_range = [2,12]

#split into train and test (90/10)
train_pct = 0.9
N_data = len(oasis_data)
N_train = int(N_data * train_pct)
N_test = N_data - N_train
range_data = np.arange(0,N_data)
train_idx = np.random.choice(range_data, N_train, replace=False)
test_idx =np.setdiff1d(range_data, train_idx)

# split the image and segmentations into the appropriate data split
oasis_data_train = oasis_data[train_idx,:]
oasis_data_test = oasis_data[test_idx,:]
oasis_data_segmentation_train = oasis_data_segmentation[train_idx,:]
oasis_data_segmentation_test = oasis_data_segmentation[test_idx,:]

# create data loaders for train and test. The GroupDataLoader will randomly sample n_input_ranges image at each iteration.
# for the test data loader, we load the entire test set.

dataset_oasis_train = GroupDataLoader(data=oasis_data_train,labels=np.zeros(N_train), class_labels=[0],
                                      segmentations=oasis_data_segmentation_train, n_inputs_range=n_inputs_range,transform=None)
dataloader_oasis_train = DataLoader(dataset_oasis_train,batch_size=1,shuffle=True)

dataset_oasis_test = SubGroupLoader(data=[oasis_data_test],labels=None, # labels=[np.zeros(N_test)],
                                     segmentations=[oasis_data_segmentation_test], transform=None)
dataloader_oasis_test = DataLoader(dataset_oasis_test, batch_size=1, shuffle=False)



# In[ ]:


img_size = torch.Size(oasis_img_size)

# image and regularization loss
criterion = losses.local_NCC_2d(volshape=img_size, lbd=1) #0.01
#criterion = losses.MinVarAndGrad2d(volshape=img_size, lbd=0.1) #0.01

# segmentation loss
seg_loss = losses.DiceWarpLoss2d(img_size)
lambda_seg = 0.5

mmnet = models.GroupNet(in_channels=1, out_channels=2, img_size=img_size).to(device)  # updated note: have vxms.models.MultiMorph now.

optimizer = optim.Adam(mmnet.parameters(), lr=0.001) #0.01


# In[ ]:


nb_epochs = 100
batch_size = 1
data_loader = dataloader_oasis_train
# move the model to the device
mmnet = mmnet.to(device)

pbar = trange(nb_epochs)
loss_hist = np.zeros(nb_epochs)

for i in pbar:
    total_running_loss = list()
    mmnet.train()
    for sample in data_loader:
        images = sample['image'].to(device)
        segmentations = sample['segmentation'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # predict the warp fields
        predw = mmnet(images)
        # compute the loss
        loss = criterion(images, predw)
        loss_seg = seg_loss(segmentations,predw)
        loss = loss + lambda_seg * loss_seg
        total_running_loss.append(loss.item())
        #optimize
        loss.backward()
        optimizer.step()

    # print stats
    m = np.mean(total_running_loss)
    pbar.set_description(f'{m:.5f}')
    loss_hist[i] = m

print('Finished Training')


# In[ ]:


# Save trained 2D model weights
import os
os.makedirs('models', exist_ok=True)
save_path = 'models/model_2d_oasis.pth'
torch.save({'state_dict': mmnet.state_dict()}, save_path)
print('Saved:', save_path)
