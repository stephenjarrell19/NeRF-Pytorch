import torch
import numpy as np
from pytorch3d.renderer import FoVPerspectiveCameras



def col_func(data):
    """
    To return FOVCam object and batched image tensor
    """
    R = []
    T = []
    fov = []
    imgs = []
    print(data[0][0].shape)
    for pt in data:
        R.append(pt[1])
        T.append(pt[2])
        fov.append(pt[-1])
        imgs.append(pt[0])
    targetCameras = FoVPerspectiveCameras(fov=fov, R=torch.tensor(R), T=torch.tensor(T))
    imgs = torch.tensor(np.array(imgs))
    return targetCameras, imgs