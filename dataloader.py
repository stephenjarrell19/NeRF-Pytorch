import json
from PIL import Image
from torch.utils.data import Dataset
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import numpy as np
import os.path as osp

class dataloader(Dataset):
    """
    Dataloader
    Returns image, Rotation, Translation and rotation matrix
    """
    def __init__(self, dataDir: str = "data/ship", dataSplit: str = "train", device: str = "cuda"):
        self.dataDir = dataDir
        self.DataFile = osp.join(dataDir, dataSplit)
        dataJson = json.load(self.DataFile)
        self.cameraAngle = dataJson["camera_angle_x"]
        self.len = len(dataJson["frames"])
        self.trainFrames = dataJson["frames"]
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        frame = self.trainFrames[idx]
        imgPath = frame["file_path"]
        img = Image.open(imgPath)
        rotation = frame["rotation"]
        P = np.array(frame["transform_matrix"])
        R = np.block(P, (0,0), (3,3))
        T = np.block(P, (0,3), (3,1))
        
        return img, R, T, rotation