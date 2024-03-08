import functools
import inspect
import logging
import os
from collections import OrderedDict
from enum import Enum
from functools import partial
from logging import Logger
from typing import List

import numpy as np
import torch
import wandb
from PIL import Image
from batch import Batch
from omegaconf import ListConfig
from pytorch_lightning.utilities import rank_zero_only
from torchvision.transforms import ToPILImage

LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))


# ======================== TRAINING UTILS ========================

class TrainStage(Enum):
    """Definition of the different training stages"""
    Training: str = "train"
    Validation: str = "valid"
    Test: str = "test"

    def is_train(self):
        """
        Checks whether the stage refers to a training stage or not
        :return: True if the stage is Training or Validation
        """
        return self == self.Training

    def __str__(self):
        return self.value


# ======================== IO UTILS ========================


def readPNG(filename):
    if not filename:
        raise ValueError("Empty filename")
    image = (np.asarray(Image.open(filename).convert("RGB")) / 255.0).astype(np.float32)
    return image


def writeEXR(data, filename):
    import imageio as imageio
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.imwrite(filename, data, flags=0x001)


def readEXR(filename):
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    if 'A' in header['channels']:
        colorChannels = ['R', 'G', 'B', 'A']
    elif 'Y' in header['channels']:
        colorChannels = ['Y']
    else:
        colorChannels = ['R', 'G', 'B']
    img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)

    return img


# ======================== LOGGING UTILS ========================


def log_anything(logger, name, data, is_metric=False, step=None, **kwargs):
    """
    Generic method to log anything
    """
    to_pil = ToPILImage()

    def prepare_data(name, data, **kwargs):
        kwargs = Batch(**kwargs)
        if isinstance(data, torch.Tensor):
            if data.ndim == 0:  # Single scalar           - Logged as point sample
                pass
            elif data.ndim == 1:  # Single vector         - Logged as histogram
                data = wandb.plot.histogram(wandb.Table(data=[[s] for s in data.cpu()], columns=["values"]), "values")
            elif data.ndim == 2:  # Single-channel image  - Logged as image
                data = wandb.Image(to_pil(data.unsqueeze(0).clamp(0, 1)), caption=kwargs.get('caption', None))
            elif data.ndim == 3:  # Multi-channel image   - Logged as image
                if data.shape[0] == 1 or data.shape[0] == 3:
                    data = wandb.Image(to_pil(data.clamp(0, 1)), caption=kwargs.get('caption', None))
                else:
                    raise NotImplementedError(f"Logging {name} in shape of {data.shape} is not implemented!")
            elif data.ndim == 4:  # Video                 - Logged as video
                data = wandb.Video((data.clamp(0, 1).cpu() * 255).to(torch.uint8),
                                   caption=kwargs.get('caption', None), format="mp4", fps=kwargs.get('fps', 4))
            else:
                raise NotImplementedError(f"Logging {name} in shape of {data.shape} is not implemented!")
            data = {name: data}
        elif isinstance(data, (list, tuple)):
            kwargs = kwargs.to_list()
            if len(kwargs) == 0:
                kwargs = [dict() for _ in range(len(data))]
            prepared_data = [prepare_data(name, d, **kwa)[name] for d, kwa in zip(data, kwargs)]
            if len(prepared_data) == 1:  # Avoid logging scalar as histogram
                prepared_data = prepared_data[0]
            data = {name: prepared_data}
        elif isinstance(data, (dict, Batch)):
            prepared_data = Batch()
            name_prefix = f"{name}/" if name is not None else ""
            for key, value in data.items():
                prepared_data.update(**prepare_data(f"{name_prefix}{key}", value, **kwargs))
            data = prepared_data.flatten(separator="/").to_dict()
        elif isinstance(data, (int, float)):
            data = {name: data}
        else:
            raise NotImplementedError(f"Logging {name} of type {type(data)} is not implemented!")

        return data

    data = prepare_data(name, data, **kwargs)
    if is_metric:
        logger.log_metrics(data, step=step)
    else:
        logger.experiment.log(data)

    return data


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

    if LOCAL_RANK != -1:
        name = f"{LOCAL_RANK}_{name}"

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


class ConsoleLogger(Logger):
    def __init__(self,
                 name,
                 id,
                 save_dir,
                 project,
                 entity,
                 plot_images=True,
                 save_images=False,
                 log_folder=None,
                 save_HDR=False,
                 **kwargs):
        super().__init__()
        self._run_name = name
        self.id = id
        self._save_dir = save_dir
        self.project = project
        self.entity = entity
        self.plot_images = plot_images
        self.save_images = save_images
        self.save_HDR = save_HDR
        self.log_folder = log_folder

        self.module_logger = init_logger()

    def get_checkpoint_path(self):
        ckpt_dir = os.path.join(self.save_dir, self.project, self.id or "", "checkpoints")

        if not os.path.exists(ckpt_dir):
            return None

        checkpoints = list(reversed(sorted(os.listdir(ckpt_dir))))
        latest_checkpoint = checkpoints[0]
        self.module_logger.info(
            f"{len(checkpoints)} checkpoints found ({checkpoints}), using the latest one: {latest_checkpoint}!")
        return os.path.join(ckpt_dir, latest_checkpoint)

    def log(self, data_dict):
        for key, data in data_dict.items():
            if isinstance(data, wandb.Image):
                self.log_image(data.image, name=key)
            elif isinstance(data, list) and isinstance(data[0], wandb.Image):
                for i, img in enumerate(data):
                    self.log_image(img.image, name=f"{key}_{i}")
            elif isinstance(data, wandb.Video):
                self.log_video(data, name=key)
            elif isinstance(data, wandb.Table):
                self.module_logger.info(f"{key}: {data.data}")
            else:
                self.module_logger.info(f"{key}: {data}")

    def log_image(self, image, name=None):
        if self.plot_images:
            image.show()
        if self.save_images:
            if self.log_folder is not None:
                img_path = os.path.join(self.log_folder, f"{name}_.png")
            else:
                img_path = os.path.join(self.save_dir, "media", "images", f"{name}_.png")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            self.module_logger.info(f"Saving image to {img_path}")
            image.save(img_path)

    def log_hdr(self, image_tensor, name=None):
        if self.save_HDR:
            if self.log_folder is not None:
                img_path = os.path.join(self.log_folder, f"{name}_.exr")
            else:
                img_path = os.path.join(self.save_dir, "media", "images", f"{name}_.exr")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            self.module_logger.info(f"Saving HDR image to {img_path}")
            writeEXR(image_tensor.cpu().permute(1, 2, 0).numpy(), img_path)

    @property
    def name(self):
        return "ConsoleLogger"

    @property
    def save_dir(self):
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return self._save_dir

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass


# ======================== CONFIG UTILS ========================


def range2list(config, max_length=None):
    if config is None:
        return list(range(max_length))
    elif isinstance(config, int):
        return [config]
    elif isinstance(config, list):
        return config
    elif isinstance(config, ListConfig):
        return config
    elif isinstance(config, str):
        slice_def = config.split(":")

        start = 0
        if len(slice_def) >= 1:
            if slice_def[0] != "":
                start = int(slice_def[0])

        end = max_length
        if len(slice_def) >= 2:
            if slice_def[1] != "":
                end = int(slice_def[1])
        assert end is not None, f"End must be defined for config {config}"

        step = 1
        if len(slice_def) == 3:
            if slice_def[2] != "":
                step = int(slice_def[2])

        return list(range(start, end, step))


# ======================== DATASTRUCTURE UTILS ========================


class LoadableObject:
    def __init__(self, load_function, val=None):
        self.load_function = load_function
        self._val = val

    def reload(self):
        self._val = self.load_function()

    @property
    def val(self):
        if self._val is None:
            self._val = self.load_function()
        return self._val


class LoadableObjectList:
    def __init__(self, lodabable_objects: List[LoadableObject]):
        self.lodabable_objects = lodabable_objects

    @property
    def val(self):
        return [lodabable_object.val for lodabable_object in self.lodabable_objects]

    def __getitem__(self, k):
        return self.lodabable_objects[k]


class LoadableObjectCache:
    def __init__(self, load_function, auto_load=True, max_size=None, name=None):
        self.name = name
        self.module_logger = init_logger()
        self.load_function = load_function
        self.max_size = max_size
        self.auto_load = auto_load
        self.cache = OrderedDict()

    def __getitem__(self, index):
        if index not in self.cache:
            # self.module_logger.debug(f"Obj cache size - {self.name}: {len(self.cache)}")
            # Remove oldest sample if max size exceeded
            if self.max_size is not None and self.max_size <= len(self.cache) and len(self.cache) > 0:
                del self.cache[list(self.cache.keys())[0]]

            # Add the new element
            loadable_object = LoadableObject(partial(self.load_function, index=index))
            if self.auto_load:
                loadable_object = loadable_object.val
            if self.max_size is not None and self.max_size > 0:
                self.cache[index] = loadable_object
        else:
            loadable_object = self.cache[index]
        return loadable_object

    def clear(self):
        self.cache = OrderedDict()


# ======================== OBJECT UTILS ========================

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if attr == '':
            return obj

        try:
            return getattr(obj, attr, *args)
        except AttributeError:
            # Try as a dictionary
            try:
                return obj[attr]
            except TypeError:
                # Try as a list
                return obj[int(attr)]

    return functools.reduce(_getattr, [obj] + attr.split('.'))

