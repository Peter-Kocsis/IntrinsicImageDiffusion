import os

import hydra
import pytorch_lightning
import torch
from batch import Batch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import ToPILImage, ToTensor, Resize, Compose

from iid.data import load_linear_image, Linear_2_SRGB, NormalizeRange, load_image
from iid.utils import init_logger, writeEXR, log_anything


def geometry_prediction(cfg: DictConfig):
    module_logger = init_logger("GeometryPrediction_MAIN")

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

    # ============= DATA =============
    # Read input
    if "input_path" in cfg.data:
        # Use single input file
        img_path = cfg.data.input_path
        module_logger.info(f"Loading input <{img_path}>")
        img = load_linear_image(img_path)

        # Prepare batch
        transforms = Compose([ToTensor(), Resize(size=[480, 640])])
        img = transforms(img).unsqueeze(0).to(device)
        original_size = img.shape[-2:]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
    else:
        # Use datamodule
        datamodule_cfg = cfg.data
        module_logger.info(f"Instantiating datamodule <{datamodule_cfg._target_}>")
        datamodule = hydra.utils.instantiate(datamodule_cfg)
        dataloader = datamodule.train_dataloader()
        assert len(dataloader) == 1, f"Only one batch should be loaded, but got {len(dataloader)} batches."
        for batch in dataloader:
            assert len(batch["im"]) == 1, f"Only a single sample can be loaded now, but got {len(batch)} samples."
            img = batch["im"].to(device)
            original_size = batch[0]["metadata"]["size"]["im"].tolist()[:2]
            image_id = batch[0]["metadata"]["sample_id"]

    img = Linear_2_SRGB()(img)  # The dataset handles the images in linear space, but OmniData expects sRGB

    # ============= MODEL =============
    # Init model
    model_cfg = cfg.model
    module_logger.info(f"Instantiating model <{model_cfg._target_}>")
    model = hydra.utils.instantiate(model_cfg)
    model = model.to(device)

    # ============= INFERENCE =============
    preds = model(img)[0]

    # ============= LOGGING =============
    # Save results
    if "output" in cfg:
        # Save to output folder
        out_folder = cfg.output.folder
        os.makedirs(out_folder, exist_ok=True)
        to_pil = ToPILImage()
        if not cfg.output.as_dataset:
            # Save with suffix
            to_pil(NormalizeRange(output_range=[0., 1.])(preds[0])).save(
                os.path.join(out_folder, f"{image_id}_depth.png"))
            to_pil((preds[1:] + 1) / 2).save(os.path.join(out_folder, f"{image_id}_normal.png"))
        else:
            # Save as dataset
            writeEXR(preds[:1].cpu().permute(1, 2, 0).numpy(), os.path.join(out_folder, "depth", f"{image_id}.exr"))
            writeEXR(preds[1:].cpu().permute(1, 2, 0).numpy(), os.path.join(out_folder, "normal", f"{image_id}.exr"))

    if "logger" in cfg:
        logger_cfg = cfg.logger
        module_logger.info(f"Instantiating logger <{logger_cfg._target_}>")
        logger = hydra.utils.instantiate(logger_cfg)

        hparams = {
            "datamodule": cfg.get("data"),
            "model": cfg.get("model"),
            "task_name": cfg.get("task_name"),
            "tags": cfg.get("tags"),
            "seed": cfg.get("seed"),
        }
        logger.log_hyperparams(hparams)

        # Log
        logs = Batch({
            "input": img[0],
            "depth_pred": NormalizeRange(output_range=[0., 1.])(preds[0]),
            "normal_pred": (preds[1:] + 1) / 2
        })
        log_anything(logger, "stage1_geometry", logs)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="stage/geometry_prediction.yaml")
def main(cfg: DictConfig):
    geometry_prediction(cfg)


if __name__ == "__main__":
    main()
