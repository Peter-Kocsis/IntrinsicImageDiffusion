import copy
import inspect
import logging
from argparse import Namespace
from collections import OrderedDict
from enum import Enum
from typing import Mapping

import numpy as np
import torch
from PIL import Image


class TrainStage(Enum):
    """Definition of the different training stages"""
    Training: str = "train"
    Validation: str = "valid"
    Test_Fit: str = "test_fit"
    Test_Eval: str = "test_eval"

    def is_train(self):
        """
        Checks whether the stage referes to a training stage or not
        :return: True if the stage is Training or Validation
        """
        return self == self.Training

    def __str__(self):
        return self.value


class IterableNamespace(Mapping, Namespace):

    def __init__(self, *args, default=None, **kwargs):
        super().__init__()
        self.__default = default
        self.__dict__.update(**kwargs)

        # Support pin memory
        if len(args) == 1 and isinstance(args[0], dict):
            self.__dict__.update(**args[0])

    def __getitem__(self, k):
        if isinstance(k, str):
            if k in self.__dict__:
                return self.__dict__.__getitem__(k)
            elif self.__default is not None:
                self.__dict__.__setitem__(k, self.__default())
                return self.__dict__.__getitem__(k)
            else:
                raise KeyError(k)
        elif isinstance(k, int):
            other = IterableNamespace()
            for key in self.keys():
                other.__dict__[key] = self.__dict__[key][k]
            return other
        elif isinstance(k, tuple):
            other = IterableNamespace()
            for key in self.keys():
                if isinstance(self.__dict__[key], (torch.Tensor, IterableNamespace)):
                    other.__dict__[key] = self.__dict__[key][k]
            return other
        raise NotImplementedError()

    def __setitem__(self, k, v):
        self.__dict__.__setitem__(k, v)
        return self

    def __delitem__(self, k):
        self.__dict__.__delitem__(k)

    def __len__(self) -> int:
        return len(list(self.keys()))

    def __iter__(self):
        for key in self.__dict__.__iter__():
            if not key.startswith("_"):
                yield key

    def keys(self):
        return (key for key in self.__dict__.keys() if not key.startswith("_"))

    def update(self, other, **kwargs):
        self.__dict__.update(other.__dict__)

    def to(self, *args, **kwargs):
        # TODO: Implement a generic method, which calls the method of all elements
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].to(*args, **kwargs)
        return other

    def repeat(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].repeat(*args, **kwargs)
        return other

    def reshape(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].reshape(*args, **kwargs)
        return other

    def permute(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].permute(*args, **kwargs)
        return other

    @property
    def ndim(self):
        ndim = None
        for key in self.keys():
            if isinstance(self.__dict__[key], (torch.Tensor, IterableNamespace)):
                if ndim is None:
                    ndim = self.__dict__[key].ndim
                else:
                    assert ndim == self.__dict__[key].ndim, "All tensors must have same number of dimensions"
        return ndim

    @property
    def shape(self):
        shape = None
        for key in self.keys():
            if isinstance(self.__dict__[key], (torch.Tensor, IterableNamespace)):
                if shape is None:
                    shape = self.__dict__[key].shape
                else:
                    assert len(shape) == len(self.__dict__[key].shape) and all(
                        (shape[idx] == self.__dict__[key].shape[idx] for idx in
                         range(len(shape)))), "All tensors must have the same shape"
        return shape

    def dim_size(self, dim=0):
        dim_size = None
        for key in self.keys():
            if isinstance(self.__dict__[key], (torch.Tensor, IterableNamespace)):
                if dim_size is None:
                    dim_size = self.__dict__[key].shape[dim]
                else:
                    other_dim_size = self.__dict__[key].shape[dim]
                    assert dim_size == other_dim_size, "All tensors must have same size"

        return dim_size

    @property
    def device(self):
        device = None
        for key in self.keys():
            if isinstance(self.__dict__[key], (torch.Tensor, IterableNamespace)):
                if device is None:
                    device = self.__dict__[key].device
                else:
                    assert device == self.__dict__[key].device, "All tensors must be on the same device"
        return device

    def unsqueeze(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in self.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].unsqueeze(*args, **kwargs)
        return other

    def gather(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].gather(*args, **kwargs)
        return other

    def index_select(self, *args, **kwargs):
        other = copy.deepcopy(self)
        for key in other.keys():
            if isinstance(other.__dict__[key], (torch.Tensor, IterableNamespace)):
                other.__dict__[key] = other.__dict__[key].index_select(*args, **kwargs)
        return other

    def cat(self, *args, dim=0, **kwargs):
        list_of_tensors = []
        cat_map = OrderedDict()
        for key in self.keys():
            if isinstance(self.__dict__[key], (torch.Tensor)):
                element = self.__dict__[key]
                list_of_tensors.append(element)
                cat_map[key] = element.shape[dim]
            if isinstance(self.__dict__[key], (IterableNamespace)):
                element, element_cat_map = self.__dict__[key].cat(*args, dim=dim, **kwargs)
                list_of_tensors.append(element)
                cat_map[key] = element_cat_map
        return torch.cat(list_of_tensors, *args, dim=dim, **kwargs), cat_map

    @classmethod
    def cat_list(cls, values, dim=0):
        other = cls()
        for key in values[0].keys():
            if isinstance(values[0].__dict__[key], torch.Tensor):
                if values[0].__dict__[key].ndim == 0:
                    other[key] = torch.stack([value.__dict__[key] for value in values])
                else:
                    other[key] = torch.cat([value.__dict__[key] for value in values], dim=dim)
            if isinstance(values[0].__dict__[key], IterableNamespace):
                other[key] = cls.cat_list([value.__dict__[key] for value in values], dim=dim)
        return other

    @classmethod
    def stack_list(cls, values, cat_if_wrong_shape=False):
        other = cls()
        for key in values[0].keys():
            if isinstance(values[0].__dict__[key], torch.Tensor):
                if cat_if_wrong_shape:
                    try:
                        other[key] = torch.stack([value.__dict__[key] for value in values])
                    except RuntimeError:
                        other[key] = torch.cat([value.__dict__[key] for value in values])[None, ...]
                else:
                    other[key] = torch.stack([value.__dict__[key] for value in values])
            if isinstance(values[0].__dict__[key], IterableNamespace):
                other[key] = cls.stack_list([value.__dict__[key] for value in values],
                                            cat_if_wrong_shape=cat_if_wrong_shape)
        return other

    @classmethod
    def from_tensor(cls, data, cat_map, dim=0):
        other = cls()
        for key, value in cat_map.items():
            if isinstance(value, int):
                other[key], data = torch.split(data, [value, data.shape[dim] - value], dim=dim)
            if isinstance(value, OrderedDict):
                other[key] = cls.from_tensor(data, value, dim=dim)
        return other


def init_logger(name: str = None, logging_level=logging.DEBUG, add_stream_handler=False) -> logging.Logger:
    """
    Creates a new logger with the given name
    :param name: The name of the logger
    :return: The created logger
    """
    if name is None:
        frame = inspect.currentframe()
        frame = frame.f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        name = local_vars["self"].__class__.__name__

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        logFormatter = logging.Formatter(fmt='%(asctime)s :: %(name)s :: %(levelname)-5s :: %(message)s')
        stream_handler.setFormatter(logFormatter)

        logger.addHandler(stream_handler)

    logger.debug(f"Logger created with name: {name}")
    return logger


def readPNG(filename):
    if not filename:
        raise ValueError("Empty filename")
    image = np.asarray(Image.open(filename).convert("RGB"))
    return image
