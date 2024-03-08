import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf

from iid.utils import init_logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="intrinsic_image_diffusion.yaml")
def main(cfg: DictConfig):
    module_logger = init_logger("IntrinsicImageDiffusion_MAIN")

    # ============= CONFIG =============

    OmegaConf.resolve(cfg)
    module_logger.info(f"Experiment config: \n{OmegaConf.to_yaml(cfg)}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        pytorch_lightning.seed_everything(cfg.seed, workers=True)

    device = cfg.device
    if device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
    module_logger.info(f"Running on {device}")

    # ============= STAGES =============
    stages = cfg.stages
    for stage_name in stages.keys():
        module_logger.info(f"=========== Running stage <{stage_name}> ===========")
        stage_cfg = OmegaConf.structured(stages[stage_name], flags={"allow_objects": True})
        if stage_cfg.get("skip") is True:
            module_logger.info(f"=========== Skipping stage <{stage_name}> ===========")
            continue

        stage_fn_cfg = stage_cfg.pop("stage_fn")
        stage_fn = hydra.utils.instantiate(stage_fn_cfg)
        stage_fn(stages[stage_name])


if __name__ == "__main__":
    main()
