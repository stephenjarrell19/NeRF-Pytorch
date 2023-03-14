import torch
import torch.nn as nn

from pytorch3d.renderer import ray_bundle_to_ray_points


class HarmonicEmbedding(nn.Module):
    def __init__(self, harmonic_dim = 10, omega = 0.1):
        super().__init__()
        self.freq = omega * (2.0 ** torch.arange(harmonic_dim))
        self.freq.requires_grad=False
        
    def forward(self, x):
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    
class NeRFModel(nn.Module):
    def __init__(self, hd_pos=10, hd_dir = 4, mlp_dim=128):
        super().__init__()
        
        self.harmonic_embedding_pos = HarmonicEmbedding(hd_pos)
        self.harmonic_embedding_dir = HarmonicEmbedding(hd_dir)
        
        emb_dim_pos = hd_pos*2*3 #(sin,cos x num_p x har_dim)
        emb_dim_dir = hd_dir*2*3 #(sin,cos x num_p x har_dim)
        
        self.mlp1 = torch.nn.Sequential(
            nn.Linear(emb_dim_pos, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
        )
        
        self.mlp2 = torch.nn.Sequential(
            nn.Linear(emb_dim_pos + mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
#             nn.ReLU(), # Model doesn't have this
            nn.Linear(mlp_dim, mlp_dim+1),
            nn.ReLU(),
        )
        
        self.mlp3 = torch.nn.Sequential(
            nn.Linear(emb_dim_dir + mlp_dim, mlp_dim/2),
            nn.ReLU(),
            nn.Linear(mlp_dim/2, 3),
            nn.Sigmoid(),
        )
        
    def forward(self, ray_bundle):
        """
        ray_bundle - B x num_rays_per_img x num_samples_per_ray x 3
        """
        
#         # B x nrays x nsam x 3
#         rays_points_world = ray_bundle_to_ray_points(ray_bundle)
#         pos_features = self.harmonic_embedding_pos(rays_points_world)
        
#         # MLP 1
#         features = self.MLP1(pos_features)
        
#         # MLP2
#         featuers = self.MLP2(torch.cat((features, pos_features),1))
#         sigma = features[:,-1]
        
#         # MLP3
#         dir_features = self.harmonic_embedding_dir(xx)
#         colors = self.MLP3(torch.cat((features, dir_features),1))
    
    

def getNeRFModel(harmonic_dim = 60, mlp_dim = 128):
    """
    Builds the MLP based NeRF model
    """
    
    # TODO
    
    
    
    