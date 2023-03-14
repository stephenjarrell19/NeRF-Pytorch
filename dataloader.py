import json
from PIL import Image
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import numpy as np
import os.path as osp

class NeRFDataset(Dataset):
    """
    Dataset class to load NeRF images.
    Returns image, Rotation, Translation and rotation matrix
    """
    def __init__(self, dataDir: str = "data/ship", dataSplit: str = "train", device: str = "cuda"):
        self.dataDir = dataDir
        self.DataFile = osp.join(dataDir, dataSplit)
        dataJson = json.load(open(osp.join(self.dataDir, "transforms_{}.json".format(dataSplit))))
        self.fov = dataJson["camera_angle_x"]
        self.len = len(dataJson["frames"])
        self.trainFrames = dataJson["frames"]
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        frame = self.trainFrames[idx]
        imgPath = osp.join(self.dataDir, frame["file_path"][2:]) + ".png"
        img = np.array(Image.open(imgPath))
        rotation = frame["rotation"]
        P = np.array(frame["transform_matrix"])
        R = P[0:3,0:3] #np.block(P, (0,0), (3,3))
        T = P[3,0:3] #np.block(P, (0,3), (3,1))
        
        # Ignoring alpha channel from input image.
        return img[:,:,:-1], R, T, rotation, self.fov