import torch
from pytorch3d.renderer.implicit.renderer import ImplicitRenderer

class Renderer(ImplicitRenderer):
    def forward(self, cameras, volumetric_function, **kwargs):
        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')
        ray_bundle = self.raysampler(
            cameras=cameras, **kwargs
        )
        
        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )

        """
        TODO
        1. Based on ray densities call the self.raysampler.forwardFine on 
        the corresponding rays and depths for fine sampling.
        2. Pass the fine samples through volumetric_function.
        3. Concatenate the fine results with images.
        """

        
        images = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs,
        )

        return images, ray_bundle