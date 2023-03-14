import torch
import torch.nn as nn


def getNeRFModel(harmonic_dim = 60, mlp_dim = 128):
    """
    Builds the MLP based NeRF model
    """
    
    