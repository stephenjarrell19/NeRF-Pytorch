from dataloader import NeRFDataset
from utils import *

import torch
from torch.utils.data import DataLoader

import numpy as np


### Set Hyper-parameters here
BATCH_SIZE = 8

# Model Params
HARMONIC_DIM = 64
MLP_DIM = 128


### Create Dataset of scene
train_dataset = DataLoader(NeRFDataset(), batch_size=BATCH_SIZE, collate_fn= col_func)


### Create renderer with ray sampler and ray marcher


### Get NeRF Model
NeRFModel = getNeRFModel(HARMONIC_DIM, MLP_DIM)

### Fit Model on the scene





