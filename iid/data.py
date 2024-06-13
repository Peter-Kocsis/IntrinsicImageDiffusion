import copy
import glob
import os
import warnings
from collections import defaultdict
from typing import Optional, Any, Callable, Union, Mapping, MutableMapping, Iterable

import hydra
import numpy as np
import torch
from batch import Batch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import RandomCrop, Compose
import torchvision.transforms.functional as F

from iid.utils import init_logger, range2list, LoadableObjectCache, readPNG, readEXR, TrainStage


# ================================ DATAMODULE ================================

class IIDDataModule(LightningDataModule):
    DEFAULT_SAMPLING_CFG = {
        "num_workers": 4,
        "batch_size": 4,
        "shuffle": True,
    }

    def __init__(self,
                 dataset_cfg: str,
                 sampling_cfg: Optional[dict] = dict()):
        super().__init__()
        self.logger = init_logger()

        self.dataset_cfg = dataset_cfg
        self.sampling_cfg = self.DEFAULT_SAMPLING_CFG
        self.sampling_cfg.update(sampling_cfg)

        self._dataset_train = None
        self._dataset_valid = None
        self._dataset_test = None

        self.sampler_type = None

    @property
    def dataset_train(self):
        if self._dataset_train is None:
            self.dataset_train = self._load_dataset(mode=TrainStage.Training)
        return self._dataset_train

    @dataset_train.setter
    def dataset_train(self, dataset_train):
        self._dataset_train = dataset_train
        self.logger.info(f"Training dataset set to {dataset_train}, number of samples: {len(dataset_train)}")

    @property
    def dataset_valid(self):
        if self._dataset_valid is None:
            self.dataset_valid = self._load_dataset(mode=TrainStage.Validation)
        return self._dataset_valid

    @dataset_valid.setter
    def dataset_valid(self, dataset_valid):
        self._dataset_valid = dataset_valid
        self.logger.info(f"Validation dataset set to {dataset_valid}, number of samples: {len(dataset_valid)}")

    @property
    def dataset_test(self):
        if self._dataset_test is None:
            self.dataset_test = self._load_dataset(mode=TrainStage.Test)
        return self._dataset_test

    @dataset_test.setter
    def dataset_test(self, dataset_test):
        self._dataset_test = dataset_test
        self.logger.info(f"Test dataset set to {dataset_test}, number of samples: {len(dataset_test)}")

    def _load_dataset(self, mode: TrainStage, **kwargs):
        self.logger.debug(f"Loading {mode} dataset! - training: {mode}")
        dataset = self.load_dataset(mode, **kwargs)
        self.logger.debug(f"The {mode} dataset loaded!")
        return dataset

    def load_dataset(self, stage: TrainStage, **kwargs):
        dataset = hydra.utils.instantiate(self.dataset_cfg,
                                          stage=stage,
                                          **kwargs)
        return dataset

    def get_dataset(self, stage):
        if stage == TrainStage.Training.value:
            dataset = self.dataset_train
        elif stage == TrainStage.Validation.value:
            dataset = self.dataset_valid
        elif stage == TrainStage.Test.value:
            dataset = self.dataset_test
        else:
            raise ValueError(f"Unknown stage {self.stage}")
        return dataset

    def prepare_data(self):
        # Called only on 1 GPU
        self.logger.debug(f"Preparing data!")
        if None in (self.dataset_train, self.dataset_test):
            raise RuntimeError("Failed ot load the dataset!")
        self.logger.debug(f"Data prepared!")

    def setup(self, stage: Optional[str] = None):
        # Called on every GPUs
        self.logger.debug(f"Setup data!")
        # Data already split into train/val/test
        self.logger.debug(f"Data set up!")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = self.dataset_train
        self.logger.debug(f"Creating train dataloader!")

        sampler = self.sampling_cfg.get("sampler", None)
        if sampler is not None:
            sampler = hydra.utils.instantiate(sampler,
                                              indices=range2list(sampler.pop("indices"),
                                                                 max_length=len(dataset)))
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=True,
            pin_memory=True,
            sampler=sampler,
            shuffle=self.sampling_cfg["shuffle"] if sampler is None else False,
        )
        self.logger.debug(
            f"Train dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = self.dataset_valid
        self.logger.debug(f"Creating validation dataloader!")
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )
        self.logger.debug(
            f"Validation dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = self.dataset_test
        self.logger.debug(f"Creating test dataloader!")
        loader = DataLoader(
            dataset,
            batch_size=self.sampling_cfg["batch_size"],
            num_workers=self.sampling_cfg["num_workers"],
            drop_last=False,
            pin_memory=True,
            shuffle=False,
        )
        self.logger.debug(
            f"Test dataloader created with dataset length: {len(loader.dataset)}, batch sampler length: {len(loader.batch_sampler) if loader.batch_sampler is not None else None}!")
        return loader


# ================================ DATASET ================================


class IIDDataset(VisionDataset):
    """
    Dataset for the Intrinsic Image Diffusion pipeline.
    It supports PNG and EXR formats, but in case of PNG, the input will be transformed to linear space
    """
    train_file = "train.txt"
    val_file = "val.txt"
    test_file = "test.txt"

    FEATURES = ["im", "albedo", "material", "normal", "depth"]
    DERIVED_FEATURES = ["shading"]

    def __init__(self,
                 root: str,
                 stage: TrainStage = TrainStage.Training,
                 features_to_include: Optional[list] = None,
                 include_metadata=True,
                 cache_size=None,
                 transform: Union[Optional[Callable], Mapping[str, Callable]] = None):
        super().__init__(root, transform=transform)
        self.module_logger = init_logger()

        self.stage = stage if isinstance(stage, TrainStage) else TrainStage(stage)
        self.features_to_include = features_to_include if features_to_include is not None else self.FEATURES
        self.include_metadata = include_metadata

        self.module_logger.debug(f"Loading {self.stage} dataset from {self.root}!")
        self.data = self.load_dataset()

        self.module_logger.debug(f"Dataset {self.stage} from {self.root} loaded (length={len(self)})!")

        self.samples = LoadableObjectCache(self._load_sample, auto_load=True, max_size=cache_size)

    @property
    def split_file_path(self) -> str:
        if self.stage == TrainStage.Training:
            return os.path.join(self.root, self.train_file)
        elif self.stage == TrainStage.Validation:
            return os.path.join(self.root, self.val_file)
        elif self.stage == TrainStage.Test:
            return os.path.join(self.root, self.test_file)
        else:
            raise ValueError(f"Invalid stage {self.stage}!")

    @property
    def split_list(self) -> list:
        # Collect the scene list
        if not os.path.exists(self.split_file_path):
            # If no split file exists, use the whole dataset for training and none for val/test
            if not any((os.path.exists(os.path.join(self.root, self.train_file)),
                        os.path.exists(os.path.join(self.root, self.val_file)),
                        os.path.exists(os.path.join(self.root, self.test_file)))):
                self.module_logger.warning(f"No split file was defined, using all data for training!")
                if self.stage == TrainStage.Training:
                    return [os.path.splitext(file_path)[0] for file_path in os.listdir(os.path.join(self.root, "im"))]
                else:
                    return []
            else:
                self.module_logger.warning(f"Split file {self.split_file_path} does not exist!")
                return []

        with open(self.split_file_path) as f:
            lines = f.readlines()
        return [line.rstrip('\n') for line in lines]

    def load_dataset(self):
        # Collect the data
        data = Batch()

        # Collect the scene list
        data['sample_ids'] = self.split_list

        # Collect the features
        self.module_logger.debug("Collecting features")
        data['samples'] = Batch(default=Batch)

        for sample_id in data['sample_ids']:
            for feature in self.features_to_include:
                data['samples'][sample_id][feature] = os.path.join(self.root, feature, sample_id)

        return data

    def __len__(self) -> int:
        return len(self.data['sample_ids'])

    def get_sample_id(self, index: int) -> str:
        try:
            return self.data['sample_ids'][index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for dataset {self.__class__.__name__} with length {len(self)}")

    def _load_sample(self, index: int) -> Any:
        sample = Batch()
        sample_id = self.get_sample_id(index)

        # Load the images
        for feature in self.features_to_include:
            image_path = os.path.join(self.root, self.data["samples"][sample_id][feature])
            sample[feature] = load_linear_image(image_path)

        # Add the metadata
        if self.include_metadata:
            sample["metadata"] = Batch()
            sample["metadata"]["sample_id"] = sample_id
            sample["metadata"]["stage"] = self.stage.value
            sample["metadata"]["size"] = Batch(
                **{feature: np.array(sample[feature].shape) for feature in self.features_to_include})

        # Transform the features
        if self.transform is not None:
            # Reset the parameters of the transform
            self.reset_transform_params(self.transform)

            # Apply different transformation to the different features
            sample = self.transform(sample)

        return sample

    def reset_transform_params(self, transform):
        if isinstance(transform, MutableMapping):
            self.reset_transform_params(list(transform.values()))
        elif isinstance(transform, Iterable):
            for t in transform:
                self.reset_transform_params(t)
        elif isinstance(transform, BatchTransform):
            for t in transform.transform.values():
                self.reset_transform_params(t)
        elif isinstance(transform, Compose):
            self.reset_transform_params(transform.transforms)
        elif hasattr(transform, "reset_parameters"):
            transform.reset_parameters()

    def __getitem__(self, index: int) -> Any:
        batch = self.samples[index]
        return batch


# ================================ TRANSFORMS ================================


class SRGB_2_Linear(object):
    def __call__(self, sample):
        return sample ** 2.2


class Linear_2_SRGB(object):
    def __call__(self, sample):
        return sample ** (1 / 2.2)


class BatchTransform(nn.Module):
    def __init__(self, transform: Union[Mapping[str, Callable], Callable], *args, **kwargs):
        super().__init__()
        self.transform = transform

    def __getitem__(self, index) -> Callable:
        if isinstance(self.transform, Mapping):
            return self.transform.get(index, self.transform.get("_default", None))
        else:
            return self.transform

    def _iter_(self):
        return iter(self.transform.values())

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Transforms the elements of a dictionary according to the transform table.
        :param x_dict: The input dictionary
        :return: The transformed dictionary
        """
        x_out = copy.copy(x_dict)
        for key, val in x_dict.items():
            transform = self[key]
            if transform is not None:
                x_out[key] = transform(val)
            else:
                x_out[key] = val

        return x_out

    def inverse(self, x_trans_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """
        Inverse transforms the elements of a dictionary according to the transform table.
        :param x_dict: The transformed dictionary
        :return: The inverse transformed dictionary
        """
        x_out = copy.deepcopy(x_trans_dict)
        for key, val in x_trans_dict.items():
            if self[key] is not None and hasattr(self[key], "inverse"):
                x_out[key] = self[key].inverse(val)
            else:
                x_out[key] = val

        return x_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"


class NanToNumTransform(torch.nn.Module):
    def __init__(self,
                 nan=0,
                 posinf=None,
                 neginf=None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def forward(self, x):
        posinf = self.posinf if self.posinf is not None else x[torch.isfinite(x)].max()
        neginf = self.neginf if self.neginf is not None else x[torch.isfinite(x)].min()

        x = torch.nan_to_num(x,
                             nan=self.nan,
                             posinf=posinf,
                             neginf=neginf)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NormalizeRange(torch.nn.Module):
    def __init__(self, output_range: list, input_range: Optional[list] = None, eps=1e-6):
        super().__init__()
        self.output_range = output_range
        self.input_range = input_range
        self.eps = eps

        self.fixed_input_range = input_range is not None

        if self.fixed_input_range:
            self.scale, self.shift = self._get_scale_shift(self.input_range, self.output_range)

    def _get_scale_shift(self, input_range, output_range):
        scale = ((output_range[1] - output_range[0]) /
                 (input_range[1] - input_range[0] + self.eps))
        shift = output_range[0] - input_range[0] * scale
        return scale, shift

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        if self.fixed_input_range:
            scale, shift = self.scale, self.shift
        else:
            input_range = x.min(), x.max()
            scale, shift = self._get_scale_shift(input_range, self.output_range)
        return x * scale + shift

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        if self.fixed_input_range:
            return (y - self.shift) / self.scale
        else:
            raise NotImplementedError("Inverse transform is not implemented for variable input range")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_range={self.input_range}, output_range={self.output_range})"


class NormalizeIntensity(torch.nn.Module):
    def __init__(self, output_mean):
        super().__init__()
        self.output_mean = output_mean

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return x / (x.mean() + 1e-6) * self.output_mean

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Intensity normalization is not invertible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_mean={self.output_mean})"


class Clamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return torch.clamp(x, min=self.min, max=self.max)

    def inverse(self, y):
        """
        Inverse transforms the range of tensor.
        :param y: The transformed tensor
        :return: The inverse transformed tensor
        """
        raise NotImplementedError(f"Clamping is not inversible")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class FixableRandomCrop(RandomCrop):
    FIXED_PARAMS = defaultdict(lambda: None)

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode="constant",
                 center_only=False,
                 fixing_id: str = None):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

        self.center_only = center_only
        self.fixing_id = fixing_id
        self.module_logger = init_logger()

    def reset_parameters(self):
        FixableRandomCrop.FIXED_PARAMS = defaultdict(lambda: None)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.center_only:
            return F.center_crop(img, self.size)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        # Get the potentially fixed parameters
        if self.fixing_id is None:
            i, j, h, w = self.get_params(img, self.size)
        else:
            if FixableRandomCrop.FIXED_PARAMS[self.fixing_id] is None:
                FixableRandomCrop.FIXED_PARAMS[self.fixing_id] = self.get_params(img, self.size)
            i, j, h, w = FixableRandomCrop.FIXED_PARAMS[self.fixing_id]

        return F.crop(img, i, j, h, w)



# ================================ IO ================================


def load_image(path, linear_space=False):
    try:
        extension = os.path.splitext(path)[1].lower()
        if extension in ['.png', '.jpg', '.jpeg']:
            image = readPNG(path)
            if linear_space:
                image = SRGB_2_Linear()(image)
            return image
        elif extension in ['.exr']:
            return readEXR(path)
    except Exception:
        warnings.warn(f"Unable to load {path}")
        raise


def load_linear_image(path):
    if path is None:
        return None

    # If full path is given, load it
    if os.path.exists(path):
        file_path = path
    else:
        # If not a full path is given, assume that extension is not defined
        file_path_list = glob.glob(f'{path}*')
        assert len(file_path_list) == 1, f"Not a single file was found for {path}: {file_path_list}"
        file_path = file_path_list[0]

    return load_image(file_path, linear_space=True)
