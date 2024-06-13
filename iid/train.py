import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from iid.utils import init_logger

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test(cfg: DictConfig):
    module_logger = init_logger("Train_MAIN")

    # ============= CONFIG =============
    OmegaConf.resolve(cfg)
    module_logger.info(f"Experiment config: \n{OmegaConf.to_yaml(cfg)}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        pytorch_lightning.seed_everything(cfg.seed, workers=True)

    # ============= DATA =============
    # Use datamodule
    datamodule_cfg = cfg.data
    module_logger.info(f"Instantiating datamodule <{datamodule_cfg._target_}>")
    datamodule = hydra.utils.instantiate(datamodule_cfg)

    # ============= MODEL =============
    # Init model
    model_cfg = cfg.model
    module_logger.info(f"Instantiating model <{model_cfg._target_}>")
    model = hydra.utils.instantiate(model_cfg)

    # ============= CALLBACKS =============
    module_logger.info("Instantiating callbacks...")
    callbacks = []
    for _, cb_conf in cfg.get("callbacks").items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            module_logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # ============= LOGGER =============
    logger_cfg = cfg.get("logger")
    module_logger.info(f"Instantiating logger <{logger_cfg._target_}>")
    logger = hydra.utils.instantiate(logger_cfg)
    if logger is not None:
        hparams = {
            "datamodule": cfg.get("data"),
            "model": cfg.get("model"),
            "task_name": cfg.get("task_name"),
            "tags": cfg.get("tags"),
            "seed": cfg.get("seed"),
        }
        logger.log_hyperparams(hparams)

    # ============= TRAINER =============
    trainer_cfg = cfg.trainer
    module_logger.info(f"Instantiating trainer <{trainer_cfg._target_}>")
    trainer: Trainer = hydra.utils.instantiate(trainer_cfg, callbacks=callbacks, logger=logger)

    # ============= TRAIN =============
    if cfg.seed is not None:
        pytorch_lightning.seed_everything(cfg.seed, workers=True)
    module_logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    test(cfg)


if __name__ == "__main__":
    main()
