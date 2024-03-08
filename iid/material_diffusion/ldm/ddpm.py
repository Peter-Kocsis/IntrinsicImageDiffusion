from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion, LatentFinetuneDiffusion, __conditioning_keys__
from ldm.util import exists, instantiate_from_config
from omegaconf import ListConfig
from torch.optim.lr_scheduler import LambdaLR


class LatentImages2ImageDiffusion(LatentFinetuneDiffusion):
    """
    condition on monocular depth estimation
    """

    def __init__(self, concat_encoding_stage_config, concat_keys=("midas_in",), *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.concat_encoding_stage_config = concat_encoding_stage_config
        self.concat_encoder = self.get_concat_encoder(concat_encoding_stage_config)
        # self.depth_stage_key = concat_keys[0]

    def get_concat_encoder(self, config):
        if not hasattr(config, "target") and config != "__is_first_stage__":
            return nn.ModuleDict(OrderedDict([(k, self.get_concat_encoder(v)) for k, v in config.items()]))

        if config == "__is_first_stage__":
            print("Using first stage also as ocncat stage.")
            return self.first_stage_model
        else:
            model = instantiate_from_config(config)
            return model

    @torch.no_grad()
    def get_cond_input(self, batch, x=None, cond_key=None, force_c_encode=False, bs=None):
        if self.model.conditioning_key is not None and not self.force_null_conditioning:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key in ['class_label', 'cls']:
                    xc = batch
                else:
                    xc = super(LatentDiffusion, self).get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if (not self.cond_stage_trainable or force_c_encode) and xc is not None:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        return c, xc

    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for depth2img'
        if isinstance(self.first_stage_key, ListConfig):
            z, c, x, xrec, xc = [], [], [], [], []
            for first_stage_key in self.first_stage_key:
                z_key, c_key, x_key, xrec_key, xc_key = super().get_input(batch, first_stage_key,
                                                                          return_first_stage_outputs=True,
                                                                          force_c_encode=True,
                                                                          return_original_cond=True, bs=bs)
                z.append(z_key)
                c.append(c_key)
                x.append(x_key)
                xrec.append(xrec_key)
                xc.append(xc_key)
            z = torch.cat(z, dim=1)
            c = torch.cat(c, dim=1)
            x = torch.cat(x, dim=1)
            xrec = torch.cat(xrec, dim=1)
            xc = torch.cat(xc, dim=1)
        else:
            z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                                  force_c_encode=True, return_original_cond=True, bs=bs)

        c_cat = self.get_cat_conditioning(batch, z.shape[2:])
        all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    def get_cat_conditioning(self, batch, shape):
        assert exists(self.concat_keys)
        # assert len(self.concat_keys) == 1
        c_cat = dict()
        for ck in self.concat_keys:
            cc = batch[ck]
            cc = rearrange(cc, 'b h w c -> b c h w')
            c_cat[ck] = cc
        c_cat = self.get_encoded_conditioning(c_cat)
        return c_cat

    def get_encoded_conditioning(self, cc, encoder=None):
        if encoder is None:
            encoder = self.concat_encoder

        if isinstance(encoder, nn.ModuleDict):
            cc = [self.get_encoded_conditioning(v, encoder=encoder[k]) for k, v in cc.items()]
            cc = torch.cat(cc, dim=1)
        else:
            if isinstance(cc, dict):
                cc = torch.cat(list(cc.values()), dim=1)

            if isinstance(encoder, AutoencoderKL):
                if cc.shape[1] == 1:
                    cc = cc.expand(-1, 3, -1, -1)
                cc = self.get_first_stage_encoding(encoder.encode(cc))
            else:
                cc = encoder(cc)
        # cc_min, cc_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3], keepdim=True)
        # cc = 2. * (cc - cc_min) / (cc_max - cc_min + 0.001) - 1.
        return cc

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        params = params + list(self.concat_encoder.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
