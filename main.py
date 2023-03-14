from dataloader import NeRFDataset
from renderer import Renderer
from sampler import Sampler
from utils import *

import torch
from torch.utils.data import DataLoader
from pytorch3d.renderer import EmissionAbsorptionRaymarcher

import numpy as np

### Set Hyper-parameters here
BATCH_SIZE = 8

# Model Params
HARMONIC_DIM = 64
MLP_DIM = 128

### Create Dataset of scene
train_dataset = DataLoader(NeRFDataset(), batch_size=BATCH_SIZE, collate_fn= col_func)


### Create renderer with ray sampler and ray marcher
volume_extent_world = 3.0
raysampler_mc = Sampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raymarcher = EmissionAbsorptionRaymarcher()

renderer_mc = Renderer(
    raysampler=raysampler_mc, raymarcher=raymarcher,
)

### Get NeRF Model
NeRFModel = getNeRFModel(HARMONIC_DIM, MLP_DIM)

### Fit Model on the scene

