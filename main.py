from dataloader import dataloader
from torch.utils.data import DataLoader
import torch
import numpy as np


train_dl = DataLoader(dataloader(), batch_size=4, collate_fn= col_func)
data_b = next(iter(train_dl))
print(data_b[1].shape)
print(data_b[0].R.shape, data_b[0].T.shape)
