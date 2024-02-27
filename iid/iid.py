"""
Based on https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/util.py
"""
from typing import Any

import torch
from pytorch_lightning import LightningModule
from einops import rearrange, repeat

from omegaconf import ListConfig

from ldm.models.diffusion.ddim import DDIMSampler

from iid.ldm.ddpm import LatentImages2ImageDiffusion
from iid.utils import IterableNamespace
from iid.utils import init_logger
from iid.utils import TrainStage


# Monkey patching for MPS support
def register_buffer(self, name, attr):
   if type(attr) == torch.Tensor:
       if attr.device != torch.device(self.model.device):
           attr = attr.float().to(torch.device(self.model.device))
   setattr(self, name, attr)
DDIMSampler.register_buffer = register_buffer


class IntrinsicImageDiffusion(LightningModule):
    """
    Intrinsic Image Diffusion model
    :param unet_config: U-Net model configuration
    :param diffusion_config: Diffusion process configuration
    :param ddim_config: DDIM sampling configuration. If not provided, DDPM is used.
    :param ckpt: Checkpoint path
    """
    def __init__(self,
                 unet_config,
                 diffusion_config,
                 ddim_config=None,
                 ckpt=None,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.module_logger = init_logger()

        self.ckpt = ckpt

        self.ddim_config = ddim_config
        self.unet_config = unet_config
        self.diffusion_config = diffusion_config

        self.diffusion_module = LatentImages2ImageDiffusion(unet_config=self.unet_config, **self.diffusion_config)

        self.num_timesteps = self.diffusion_config.timesteps

        if self.ckpt is not None:
            self.init_from_ckpt(self.ckpt)

        self.shape = self.get_shape()

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_shape(self):
        image_size = self.diffusion_module.image_size \
            if hasattr(self.diffusion_module.image_size, "__getitem__") \
            else [self.diffusion_module.image_size, self.diffusion_module.image_size]
        shape = (self.diffusion_module.channels, image_size[0], image_size[1])
        return shape

    def _nan_to_num(self, conditioning_img):
        if torch.isfinite(conditioning_img).sum() > 0:
            max_val = conditioning_img[torch.isfinite(conditioning_img)].max()
            min_val = conditioning_img[torch.isfinite(conditioning_img)].min()
        else:
            max_val = 0
            min_val = 0
        return torch.nan_to_num(conditioning_img, nan=0, posinf=max_val, neginf=min_val)

    def encode(self, x):
        if isinstance(x, IterableNamespace):
            z = []
            for key in x.keys():
                z_key = self.encode(x[key])
                z.append(z_key)
            z = torch.cat(z, dim=1)
            return z
        else:
            return self.diffusion_module.get_first_stage_encoding(self.diffusion_module.encode_first_stage(x))

    def decode(self, z):
        x = IterableNamespace()
        if isinstance(self.diffusion_config.first_stage_key, ListConfig):
            z_s = torch.split(z, z.shape[1] // len(self.diffusion_config.first_stage_key), dim=1)
            for key, z_key in zip(self.diffusion_config.first_stage_key, z_s):
                x[key] = self.diffusion_module.decode_first_stage(z_key)
        else:
            x[self.diffusion_config.first_stage_key] = self.diffusion_module.decode_first_stage(z)
        return x

    @torch.no_grad()
    def sample(self,
               conditioning_img,
               batch_size=16,
               return_intermediates=False,
               x_T=None,
               **kwargs):
        """
        Sample from the model
        :param conditioning_img: Image tensor of shape (1, channels, height, width) in the range of [0, 1]
        :param batch_size: Batch size
        :param return_intermediates: Whether to return intermediate samples
        :param x_T: Noise at timestep T
        :param kwargs: Additional arguments
        :return: Sampled image tensor of shape (batch_size, channels, height, width) in the range of [0, 1]
        """
        # Prepare the conditioning
        conditioning = IterableNamespace()
        conditioning.im = self._nan_to_num(conditioning_img * 2 - 1).float()

        # Sampling logic
        for k in conditioning.keys():
            conditioning[k] = rearrange(conditioning[k], 'b c h w -> b h w c')

        c, _ = self.diffusion_module.get_cond_input(conditioning)

        if conditioning.shape[0] != batch_size:
            conditioning = conditioning.repeat(batch_size, *([1] * (conditioning.ndim - 1)))
            c = c.repeat(batch_size, *([1] * (c.ndim - 1))) if c is not None else c
        conditioning = self.diffusion_module.get_cat_conditioning(conditioning, shape=(
            self.diffusion_module.image_size, self.diffusion_module.image_size))

        uncond_conditioning = None
        unconditional_guidance_scale = 1
        if c is not None:
            conditioning = {"c_concat": [conditioning], "c_crossattn": [c]}
        else:
            conditioning = conditioning

        # Run the sampling
        if self.ddim_config is not None:
            ddim_model = DDIMSampler(self.diffusion_module)
            z, z_intermediates = ddim_model.sample(S=self.ddim_config.S,
                                                   conditioning=conditioning,
                                                   batch_size=batch_size,
                                                   x_T=x_T,
                                                   eta=self.ddim_config.eta,
                                                   unconditional_guidance_scale=unconditional_guidance_scale,
                                                   unconditional_conditioning=uncond_conditioning,
                                                   shape=self.shape,
                                                   verbose=False,
                                                   **kwargs)
        else:
            z, z_intermediates = self.diffusion_module.sample(cond=conditioning,
                                                              batch_size=batch_size,
                                                              return_intermediates=return_intermediates,
                                                              shape=self.shape,
                                                              x_T=x_T)

        # Decode the samples
        y = self.decode(z)
        y = torch.cat((y['albedo'], y['material']), dim=1)
        material = (y + 1) / 2

        if return_intermediates:
            x_intermediates = (self.decode(z_intermediates)['albedo'] + 1) / 2
            return material, x_intermediates
        return material
