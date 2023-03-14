import torch
from pytorch3d.renderer.implicit.renderer import ImplicitRenderer

class Renderer(ImplicitRenderer):
    def forward(self, cameras, volumetric_function, **kwargs):
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).

        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.

        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `Union[RayBundle, HeterogeneousRayBundle]` containing
                the parametrizations of the sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        # first call the ray sampler that returns the RayBundle or HeterogeneousRayBundle
        # parametrizing the rendering rays.
        ray_bundle = self.raysampler(
            cameras=cameras, **kwargs
        )
        # ray_bundle.origins - minibatch x ... x 3
        # ray_bundle.directions - minibatch x ... x 3
        # ray_bundle.lengths - minibatch x ... x n_pts_per_ray
        # ray_bundle.xys - minibatch x ... x 2

        # given sampled rays, call the volumetric function that
        # evaluates the densities and features at the locations of the
        # ray points
        # pyre-fixme[23]: Unable to unpack `object` into 2 values.
        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        """
        TODO
        1. Based on ray densities call the self.raysampler.forwardFine on 
        the corresponding rays and depths for fine sampling.
        2. Pass the fine samples through volumetric_function.
        3. Concatenate the fine results with images.
        """

        # finally, march along the sampled rays to obtain the renders
        images = self.raymarcher(
            rays_densities=rays_densities,
            rays_features=rays_features,
            ray_bundle=ray_bundle,
            **kwargs,
        )
        # images - minibatch x ... x (feature_dim + opacity_dim)

        return images, ray_bundle