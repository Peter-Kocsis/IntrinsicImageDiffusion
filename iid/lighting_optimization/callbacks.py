import copy
import os
import shutil
from abc import ABC
from collections import defaultdict
from typing import Any, Optional, Dict, Tuple

import numpy as np
import pytorch_lightning
import torch
import wandb
from batch import Batch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import Identity
from torchvision.transforms import ToPILImage, Resize
from torchvision.transforms.functional import resize

from iid.data import Linear_2_SRGB
from iid.lighting_optimization.pruning import ThresholdPruning
from iid.lighting_optimization.render import IIR_SSRT_RenderLayer
from iid.utils import rgetattr, LOCAL_RANK, init_logger, log_anything


class ScheduledCallback(ABC, pytorch_lightning.Callback):
    """
    Abstract callback that is executed on a schedule.
    By default, the callback is scheduled for every train epoch start.
    :param log_schedule: A dictionary that maps the name of the callback method to a schedule.
                         The schedule can be
                           - single integer
                           - a list of integers
                           - a string that defines a slice
                           - a dictionary that defines a condition with a dynamic value
                           - boolean
    :param rank_zero_only: If True, the callback is only executed on the rank 0 process.
    """
    def __init__(self,
                 log_schedule=None,
                 rank_zero_only=False):
        self.log_schedule = defaultdict(lambda: None)
        self.rank_zero_only = rank_zero_only
        if log_schedule is not None:
            self.log_schedule.update(log_schedule)
        else:
            self.log_schedule["on_train_epoch_start"] = "::1"

    def should_log(self, key, trainer):
        if trainer.logger is None:
            return False

        schedule = self.log_schedule[key]
        if schedule is None:
            return False

        if isinstance(schedule, bool):
            return schedule

        if isinstance(schedule, dict):
            # Run only if the defined dynamic value has the given value
            for dynamic_name, value in schedule.items():
                return rgetattr(trainer, dynamic_name) == value

        if "epoch" in key:
            current_val = trainer.current_epoch
            max_val = trainer.max_epochs
        elif "batch" in key:
            current_val = trainer.global_step - 1
            max_val = None  # Not implemented
        else:
            raise NotImplementedError(f"Unknown key {key}")

        if isinstance(schedule, int):
            if schedule < 0:
                schedule = max_val - 1 - schedule - 1
            return current_val == schedule
        elif isinstance(schedule, list):
            return current_val in schedule
        elif isinstance(schedule, str):
            slice_def = schedule.split(":")

            start = 0
            if len(slice_def) >= 1:
                if slice_def[0] != "":
                    start = int(slice_def[0])

            end = None
            if len(slice_def) >= 2:
                if slice_def[1] != "":
                    end = int(slice_def[1])

            step = 1
            if len(slice_def) == 3:
                if slice_def[2] != "":
                    step = int(slice_def[2])

            if end is None:
                return current_val >= start and (current_val - start) % step == 0
            else:
                return current_val in range(start, end, step)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""
        if not self.rank_zero_only or LOCAL_RANK == -1:
            if self.should_log("on_train_epoch_start", trainer):
                self(datamodule=trainer.datamodule, logger=trainer.logger, pl_module=pl_module)

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        if not self.rank_zero_only or LOCAL_RANK == -1:
            if self.should_log("on_train_batch_start", trainer):
                self(datamodule=trainer.datamodule, logger=trainer.logger, pl_module=pl_module, batch=batch)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        """Called when the train epoch begins."""
        if not self.rank_zero_only or LOCAL_RANK == -1:
            if self.should_log("on_train_batch_end", trainer):
                self(datamodule=trainer.datamodule, logger=trainer.logger, pl_module=pl_module, outputs=outputs,
                     batch=batch)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""
        if not self.rank_zero_only or LOCAL_RANK == -1:
            if self.should_log("on_fit_end", trainer):
                self(datamodule=trainer.datamodule, logger=trainer.logger, pl_module=pl_module)

    def __call__(self, datamodule, logger, pl_module, outputs=None, batch=None):
        pass


class FileCopy(ScheduledCallback):
    """
    Callback to copy files
    """
    def __init__(self, src, dst, log_schedule=None):
        super().__init__(log_schedule)
        self.src = src
        self.dst = dst

        self.module_logger = init_logger()

    @torch.no_grad()
    def __call__(self, datamodule, logger, pl_module, outputs=None, batch=None):
        # Copy the file
        os.makedirs(os.path.dirname(self.dst), exist_ok=True)
        shutil.copy(self.src, self.dst)
        self.module_logger.info(f"Copied {self.src} to {self.dst}")


class PredictionLogger(ScheduledCallback):
    """"
    Callback to log predictions

    """
    def __init__(self,
                 keys_to_tonemap=None,
                 lighting_transforms=None,
                 keys_to_log=None,
                 context="predictions",
                 eval_resolution=None,
                 log_schedule=None):
        super().__init__(log_schedule)
        self.module_logger = init_logger()
        self.lighting_transforms = lighting_transforms
        self.keys_to_log = keys_to_log
        self.context = context
        self.eval_resolution = eval_resolution

        self.keys_to_tonemap = keys_to_tonemap

        if self.keys_to_tonemap is not None:
            # self.tone_mapping = DeepHDRToneMapping()
            self.tone_mapping = Linear_2_SRGB()
        else:
            self.tone_mapping = Identity()

        self.to_pil = ToPILImage()

    @torch.no_grad()
    def __call__(self, datamodule, logger, pl_module, outputs=None, batch=None, name_prefix=""):
        # Prepare the batch
        if batch is None:
            batch = next(iter(datamodule.train_dataloader()))
        batch = Batch(**batch)

        # Prepare the output
        if self.eval_resolution is None:
            if outputs is None:
                batch = batch.to(pl_module.device)

                outputs = pl_module(batch)
                if self.lighting_transforms is not None:
                    for lighting_transform in self.lighting_transforms:
                        lighting_transform(pl_module.lighting_model)
                        outputs_relight = pl_module(batch)

                    relit_image = self.residual_editing(src=outputs["pred"].clip(0, 1),
                                                        tgt=outputs_relight["pred"].clip(0, 1),
                                                        im=outputs["target"].clip(0, 1))
                    outputs["pred_relit"] = outputs_relight.pred
                    outputs["relit"] = relit_image
            else:
                outputs = Batch(**outputs)
        else:
            batch = batch.to(pl_module.device).copy()
            batch = batch.map(resize, self.eval_resolution)

            renderer = pl_module.renderer
            pl_module.renderer = IIR_SSRT_RenderLayer(imWidth=self.eval_resolution[1],
                                                      imHeight=self.eval_resolution[0]).to("cuda")
            outputs = pl_module(batch)
            pl_module.renderer = renderer

        # Add envmap to logs
        outputs["envmap"] = self.get_envmap(pl_module)

        # Collect
        summary = Batch(
            input=batch,
            output=outputs
        )
        summary = summary.flatten()
        summary = summary[self.keys_to_log]

        summary[self.keys_to_tonemap] = summary[self.keys_to_tonemap].map(self.tone_mapping)

        summary = summary.map(list)  # List of images are handled as different images. 4D tensor is handled as video
        log_anything(logger=logger, name=self.context, data=summary)

    def get_envmap(self, pl_module):
        # Prepare directions
        envWidth, envHeight = 640, 320

        Az = ((np.arange(envWidth) + 0.5) / envWidth - 0.5) * 2 * np.pi
        El = ((np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        directions = np.concatenate((lx, ly, lz), axis=0)
        directions = torch.tensor(directions, device="cuda", dtype=torch.float32).permute(1, 2, 0).reshape(1, -1, 3)

        # Evlauate the lighting
        lighting_model = pl_module.lighting_model.lightings["envmap"]
        lighting_image = lighting_model(directions)
        lighting_image = lighting_image / (lighting_image.mean() + 1e-6) * 0.5
        lighting_image = lighting_image.reshape(1, envHeight, envWidth, 3).permute(0, 3, 1, 2)

        lighting_image = lighting_image.clip(0, 1)

        return lighting_image

    def residual_editing(self, src, tgt, im):
        # Calculate residual
        res_edit = tgt / src
        res_edit[src.expand_as(tgt) == 0] = 1

        # Match the resolution
        res_edit = Resize(im.shape[-2:])(res_edit)

        # Apply the residual
        im = im * res_edit

        return im


class LearningRateChangeMonitor(LearningRateMonitor):
    def __init__(self, logging_interval: Optional[str] = None, log_momentum: bool = False) -> None:
        super().__init__(logging_interval, log_momentum)
        self.prev_stat = defaultdict(lambda: None)

    def _extract_stats(self, trainer: "pl.Trainer", interval: str) -> Dict[str, float]:
        latest_stat = {}

        (
            scheduler_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(trainer.lr_scheduler_configs)
        self._remap_keys(scheduler_hparam_keys)

        for name, config in zip(scheduler_hparam_keys, trainer.lr_scheduler_configs):
            if interval in [config.interval, "any"]:
                opt = config.scheduler.optimizer
                current_stat = self._get_lr_momentum_stat(opt, name)
                latest_stat.update(current_stat)

        optimizer_hparam_keys, optimizers_without_scheduler = self._find_names_from_optimizers(
            trainer.optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        self._remap_keys(optimizer_hparam_keys)

        for opt, names in zip(optimizers_without_scheduler, optimizer_hparam_keys):
            current_stat = self._get_lr_momentum_stat(opt, names)
            latest_stat.update(current_stat)

        latest_stat = self._filter_stats(latest_stat)

        return latest_stat

    def _filter_stats(self, latest_stat):
        filtered_stat = {}

        for key, value in latest_stat.items():
            if self.prev_stat[key] != value:
                filtered_stat[key] = value
                self.prev_stat[key] = value
        return filtered_stat


class IterativeLightingPruning(ScheduledCallback):
    def __init__(self,
                 module_name=None,
                 param_name=None,
                 rel_threshold=0.2,
                 exp_threshold=True,
                 context="stats",
                 log_schedule=None):
        super().__init__(log_schedule)
        self.module_name = module_name
        self.param_name = param_name

        self.rel_threshold = rel_threshold
        self.exp_threshold = exp_threshold
        self.context = context

        self.module_logger = init_logger()

    @torch.no_grad()
    def __call__(self, datamodule, logger, pl_module, outputs=None, batch=None):
        # Collect the parameters to prune
        parameters_to_prune, parameters = self.collect_parameters(pl_module, self.module_name, self.param_name)

        # Determine the threshold
        max_val = torch.tensor(-torch.inf)
        for name, param in parameters:
            max_val = torch.maximum(max_val.to(param.device), torch.max(param))

        if self.exp_threshold:
            threshold = torch.log(torch.tensor(self.rel_threshold, device=max_val.device)) + max_val
        else:
            threshold = self.rel_threshold * max_val

        # Prune the parameters
        for layer_to_prune, param_name in parameters_to_prune:
            ThresholdPruning.apply(layer_to_prune, name=param_name, threshold=threshold)

        self.log_sparsity_stats(logger, parameters_to_prune)

    def collect_parameters(self, module, module_name, param_name):
        # Get root module
        root_module = module
        if module_name is not None:
            root_module = rgetattr(module, module_name)

        # Get all parameters
        parameters_to_prune = list(root_module.named_parameters())
        if param_name is not None:
            parameters_to_prune = [p for p in parameters_to_prune if param_name in p[0]]

        # Get the corresponding layers
        layers_to_prune = [(rgetattr(root_module, ".".join(name.split(".")[:-1])), name.split(".")[-1]) for name, _ in
                           parameters_to_prune]
        return layers_to_prune, parameters_to_prune

    @rank_zero_only
    def log_sparsity_stats(self, logger, parameters_to_prune) -> None:
        curr = [self.get_pruned_stats(m) for m, _ in parameters_to_prune]
        curr = torch.cat(curr)
        curr_total_enabled = curr.sum()

        total = len(curr)
        curr_total_disabled = total - curr_total_enabled

        self.module_logger.info(
            f"Total pruned:"
            f" {curr_total_disabled}/{total} ({curr_total_disabled / total:.2%})"
        )

        logger.experiment.log({f"{self.context}/total_lamps": curr_total_enabled})

    @staticmethod
    def get_pruned_stats(module: nn.Module) -> Tuple[int, int]:
        return module.is_enabled
