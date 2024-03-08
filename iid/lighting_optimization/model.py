import copy
import re
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from batch import Batch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Optimizer
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.transforms import ToPILImage

from iid.data import Linear_2_SRGB
from iid.lighting_optimization.render import IIR_SSRT_RenderLayer, depth_to_vpos
from iid.utils import init_logger, TrainStage


# ======================== MODULE ========================


class EmissiveLightingModel(LightningModule):
    """
    Lighting module
    """

    def __init__(self,
                 lighting_model,
                 renderer_args,
                 loss_cfg=None,
                 optimizer=None,
                 scheduler=None):
        super().__init__()
        self.module_logger = init_logger()

        self.lighting_model = lighting_model
        self.loss_cfg = defaultdict(float)
        if loss_cfg is not None:
            self.loss_cfg.update(loss_cfg)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # self.fov = 60
        self.fov = 100

        self.rendering_loss = PartiallyClampedMSELoss()
        self.renderer = IIR_SSRT_RenderLayer(**renderer_args, fov=self.fov)

        self.psnr = PeakSignalNoiseRatio()
        self.tone_mapping = Linear_2_SRGB()

    def forward(self, batch, lighting_model=None):
        if lighting_model is None:
            lighting_model = self.lighting_model

        batch = copy.deepcopy(batch)

        input_batch = batch
        input_batch['roughness'] = input_batch.material[:, :1, ...]
        input_batch['metallic'] = input_batch.material[:, 1:2, ...]

        input_batch['depth'] = (input_batch['depth'] + 1) / 2

        target = batch.im

        # Predict
        model_out = self.render(lighting_model=lighting_model, material=input_batch, geometry=input_batch, image=target)
        pred = model_out.rendering

        batch_info = model_out
        batch_info["pred"] = pred
        batch_info["target"] = target

        return batch_info

    def general_step(self, batch, batch_idx, mode: TrainStage):
        """
        General step used in all phases
        :param batch: The current batch of data
        :param batch_idx: The current batch index
        :param mode: The current phase
        :return: The loss
        """
        # Setup hooks
        inspected_variables = {}
        hooks = []
        # hooks.append(inspect_layer_output(model=self,
        #                                   layer_path="conditioning",
        #                                   storage_dict=inspected_variables,
        #                                   unsqueeze=False))

        # ======================== STEP ========================
        batch_info = self(batch)
        pred = batch_info.pred
        target = batch_info.target

        # Remove hook handlers
        for hook in hooks:
            hook.remove()

        # Calculate loss
        loss_info = self.calc_loss(pred=pred, target=target, lighting_model=self.lighting_model)

        # Log metrics
        for loss_name, loss_val in loss_info.items():
            self.log(f'loss/{mode}_{loss_name}', loss_val, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        batch_info["loss"] = loss_info.loss

        return dict(batch_info)

    def calc_loss(self, pred, target, lighting_model):
        loss_info = Batch()

        # Rerendering loss
        loss_info.rendering_loss = self.rendering_loss(pred, target)

        # Regularization
        loss_info.val_reg = lighting_model.val_reg_loss()
        loss_info.pos_reg = lighting_model.pos_reg_loss()

        loss_info.loss = (loss_info.rendering_loss +
                          self.loss_cfg["w_val_reg"] * loss_info.val_reg +
                          self.loss_cfg["w_pos_reg"] * loss_info.pos_reg)

        # Add PSNR eval
        loss_info.psnr = self.psnr(self.tone_mapping(pred.clamp(0, 1)),
                                   self.tone_mapping(target.clamp(0, 1)))

        return loss_info

    def render(self, lighting_model, material, geometry, image=None):
        albedo = material.albedo.clamp(0, 1)
        rough = material.roughness.clamp(0, 1)
        metal = material.metallic.clamp(0, 1)

        normal = geometry.normal
        depth = geometry.depth

        vpos = depth_to_vpos(torch.clamp(depth[0, 0], min=1e-6), self.fov, True)
        vpos = vpos.unsqueeze_(0)

        lighting_model.position_init(vpos, normal, image)

        colorDiffuse, colorSpec, wi_mask, shading = self.renderer(lighting_model=lighting_model,
                                                         albedo=albedo,
                                                         rough=rough,
                                                         metal=metal,
                                                         normal=normal,
                                                         vpos=vpos)
        rendering = colorDiffuse + colorSpec
        rendering = torch.nan_to_num(rendering, nan=0.0, posinf=0.0, neginf=0.0)

        to_pil = ToPILImage()

        # shading = shading / 101

        rendering_results = Batch()
        rendering_results["colorDiffuse"] = colorDiffuse
        rendering_results["colorSpec"] = colorSpec
        rendering_results["wi_mask"] = wi_mask
        rendering_results["shading"] = shading / (shading.mean() + 1e-6) * 0.5
        rendering_results["rendering"] = rendering

        return rendering_results

    def training_step(self, batch, batch_idx, *args):
        """Abstract definition of the training step"""
        return self.general_step(batch, batch_idx, TrainStage.Training)

    def validation_step(self, batch, batch_idx):
        """Abstract definition of the validation step"""
        return self.general_step(batch, batch_idx, TrainStage.Validation)

    def test_step(self, batch, batch_idx):
        """Abstract definition of the test step"""
        raise NotImplementedError()

    def configure_optimizers(self):
        if isinstance(self.optimizer, dict):
            optimizers = dict()
            for optimizer_pattern, optimizer in self.optimizer.items():
                optimizer_params = [param for param_name, param in self.named_parameters() if len(re.findall(optimizer_pattern, param_name)) > 0]
                param_optimizer = optimizer(params=optimizer_params)
                # param_scheduler = self.configure_scheduler((optimizer_pattern, param_optimizer))
                # optimizer_config.append({"optimizer": optimizer, **param_scheduler})
                optimizers[optimizer_pattern] = param_optimizer
            optimizer = PerParameterGroupOptimizer(**optimizers)
            param_scheduler = self.configure_scheduler(optimizer)

            optimizer_config = {"optimizer": optimizer, **param_scheduler}
            self.module_logger.info(f"Optimizer: {optimizer_config}")
            return optimizer_config
        else:
            optimizer = self.optimizer(params=self.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "loss/train_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}

    def configure_scheduler(self, optimizer):
        if self.scheduler is None:
            return {}

        if isinstance(optimizer, tuple):
            if isinstance(self.scheduler, dict):
                scheduler = self.scheduler[optimizer[0]](optimizer=optimizer[1])
            else:
                scheduler = self.scheduler(optimizer=optimizer[1])
        else:
            scheduler = self.scheduler(optimizer=optimizer)

        return {
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ======================== LOSS ========================


class PartiallyClampedMSELoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        clamp_mask = torch.any(target > 0.90, dim=1, keepdim=True).expand_as(target)

        input[clamp_mask] = input[clamp_mask].clamp(0, 1)
        return super().forward(input, target)


# ======================== OPTIMIZER ========================


class PerParameterGroupOptimizer(Optimizer):
    r"""
    Implements a compound optimizer, consisting of multiple optimizers, but using a single forwards pass
    """

    def __init__(self, **optimizers):
        self.optimizers = dict(optimizers)
        params = []
        defaults = {}
        for optimizer_pattern, optimizer in self.optimizers.items():
            params.extend(optimizer.param_groups)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers.values():
            optimizer.step()

        return loss

    def __setstate__(self, state):
        for optimizer_pattern, optimizer in self.optimizers.items():
            optimizer.__setstate__(state[optimizer_pattern])

    def __repr__(self):
        repr_str = ""
        for optimizer_pattern, optimizer in self.optimizers.items():
            repr_str += f"Optimizer {optimizer_pattern}\n{optimizer}"
        return repr_str

    def state_dict(self) -> dict:
        states = dict()

        for optimizer_pattern, optimizer in self.optimizers.items():
            optimizer_state_dict = optimizer.state_dict()
            states[optimizer_pattern] = optimizer_state_dict

        return states

    def load_state_dict(self, state_dict: dict) -> None:
        for optimizer_pattern, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[optimizer_pattern])

    def zero_grad(self, set_to_none: Optional[bool] = ...) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none)

    def add_param_group(self, param_group: dict) -> None:
        super(PerParameterGroupOptimizer, self).add_param_group(param_group)
