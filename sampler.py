import torch
from pytorch3d.renderer import MonteCarloRaysampler

class Sampler(torch.nn.Module):
    def __init__(self, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, n_rays_per_image=720, n_pts_per_ray=128, min_depth=0.1, max_depth=3.0):
        self.coarseMcSampler = MonteCarloRaysampler(
            min_x = min_x,
            max_x = max_x,
            min_y = min_y,
            max_y = max_y,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=min_depth,
            max_depth=max_depth
        )

        self.fineMinDepth = 0.1
        self.fineMaxDepth = 3.0

        self.fineMcSampler = MonteCarloRaysampler(
            min_x = min_x,
            max_x = max_x,
            min_y = min_y,
            max_y = max_y,
            n_rays_per_image=n_rays_per_image,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=self.fineMinDepth,
            max_depth=self.fineMaxDepth
        )

    def forward(self, cameras, **kwargs):
        return self.coarseMcSampler(cameras)
    
    def forwardFine(self, cameras, **kwargs):
        pass
