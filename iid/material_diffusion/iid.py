"""
Based on https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/util.py
"""
import warnings
from collections import OrderedDict
from typing import Any, List, Mapping

import torch
from batch import Batch
from ldm.util import exists
from pytorch_lightning import LightningModule
from einops import rearrange

from omegaconf import ListConfig

from ldm.models.diffusion.ddim import DDIMSampler
from torch.nn.modules.module import _IncompatibleKeys

from iid.material_diffusion.ldm.ddpm import LatentImages2ImageDiffusion
from iid.utils import init_logger, TrainStage


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
                 learning_rate=1e-5,
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

        self.diffusion_module.learning_rate = learning_rate

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
        if isinstance(x, Batch):
            z_batch = x.map(self.encode)
            z = torch.cat([val for _, val in z_batch.items()], dim=1)
            return z
        else:
            return self.diffusion_module.get_first_stage_encoding(self.diffusion_module.encode_first_stage(x))

    def decode(self, z):
        x = Batch()
        if isinstance(self.diffusion_config.first_stage_key, ListConfig):
            z_s = Batch.from_tensor(z,
                                    {key: z.shape[1] // len(self.diffusion_config.first_stage_key)
                                     for key in self.diffusion_config.first_stage_key},
                                    dim=1,
                                    split_fn=torch.split)
            x = z_s.map(self.diffusion_module.decode_first_stage)
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
        conditioning = Batch()
        conditioning.im = self._nan_to_num(conditioning_img * 2 - 1).float()

        # Sampling logic
        conditioning.map(rearrange, pattern='b c h w -> b h w c')
        for k in conditioning.keys():
            conditioning[k] = rearrange(conditioning[k], 'b c h w -> b h w c')

        c, _ = self.diffusion_module.get_cond_input(conditioning)

        if conditioning.im.shape[0] != batch_size:
            conditioning.im = conditioning.im.repeat(batch_size, *([1] * (conditioning.im.ndim - 1)))
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
            z = self.diffusion_module.sample(cond=conditioning,
                                            batch_size=batch_size,
                                            return_intermediates=return_intermediates,
                                            x_T=x_T)
            if return_intermediates:
                z, z_intermediates = z

        # Decode the samples
        y = self.decode(z)
        y = torch.cat((y['albedo'], y['material']), dim=1)
        material = (y + 1) / 2

        if return_intermediates:
            x_intermediates = (self.decode(z_intermediates)['albedo'] + 1) / 2
            return material, x_intermediates
        return material

    # ============================ TRAINING ============================
    def configure_optimizers(self):
        return self.diffusion_module.configure_optimizers()

    def general_step(self, batch, batch_idx, mode: TrainStage):
        """
        General step used in all phases
        :param batch: The current batch of data
        :param batch_idx: The current batch index
        :param mode: The current phase
        :return: The loss
        """
        # Workaround:
        batch = Batch(**batch)

        # ======================== STEP ========================
        # Compose StableDiffusion batch
        batch = self.prepare_batch(batch)

        loss, loss_dict = self.diffusion_module.shared_step(batch)
        loss_info = Batch()
        loss_info.loss = loss
        loss_info.loss_simple = loss_dict['train/loss_simple']
        loss_info.loss_elbo = 0

        return loss_info.loss

    def training_step(self, batch, batch_idx, *args):
        """Abstract definition of the training step"""
        return self.general_step(batch, batch_idx, TrainStage.Training)

    def prepare_batch(self, batch):
        batch = batch.map(rearrange, pattern='b c h w -> b h w c')
        return batch

    def get_input_from_batch(self, batch):
        prepared_batch = self.prepare_batch(batch)
        z, c = self.diffusion_module.get_input(prepared_batch, self.diffusion_module.first_stage_key)
        return z, c

    def get_conditioning_from_batch(self, batch):
        prepared_batch = self.prepare_batch(batch)
        assert exists(self.diffusion_module.concat_keys), "Concat keys must be provided"
        assert len(self.diffusion_module.concat_keys) == 1, "Only one conditioning key is supported"
        c_cat = batch[list(self.diffusion_module.concat_keys)[0]]
        return c_cat

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # Try to load, but keep if no success!
            try:
                module._load_from_state_dict(
                    local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            except RuntimeError as e:
                if strict:
                    raise e
                else:
                    error_msgs.append(e)
                    return
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            if strict:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                   self.__class__.__name__, "\n\t".join(error_msgs)))
            else:
                warnings.warn('Error(s) in loading state_dict for {}:\n\t{}'.format(
                              self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)
