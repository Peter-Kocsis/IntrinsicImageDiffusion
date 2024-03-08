import os

import hydra
import pytorch_lightning
import torch
from batch import Batch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose

from iid.data import load_linear_image, Linear_2_SRGB, SRGB_2_Linear
from iid.material_diffusion.iid import IntrinsicImageDiffusion
from iid.utils import init_logger, writeEXR, log_anything


def material_diffusion(cfg: DictConfig):
    module_logger = init_logger("MaterialDiffusion_MAIN")

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
        assert len(dataloader) == 1, "Only one batch should be loaded"
        for batch in dataloader:
            assert len(batch["im"]) == 1, "Only a single sample can be loaded now"
            img = batch["im"].to(device)
            original_size = batch[0]["metadata"]["size"]["im"].tolist()[:2]
            image_id = batch[0]["metadata"]["sample_id"]

    # ============= MODEL =============
    # Load config
    config = OmegaConf.load(cfg.model.config_path)

    # Init model
    model = IntrinsicImageDiffusion(unet_config=config.unet_config,
                                    diffusion_config=config.diffusion_config,
                                    ddim_config=config.ddim_config)

    # Load weights
    module_logger.info(f"Loading model <{cfg.model.ckpt_path}>")
    ckpt = torch.load(cfg.model.ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # ============= INFERENCE =============
    preds = predict_materials(model, img,
                              num_samples=cfg.model.num_samples,
                              sampling_batch_size=cfg.model.sampling_batch_size,
                              original_size=original_size)

    # ============= LOGGING =============
    # Save results
    if "output" in cfg:
        # Save to output folder
        out_folder = cfg.output.folder
        os.makedirs(out_folder, exist_ok=True)

        to_pil = ToPILImage()
        if not cfg.output.as_dataset:
            # Save with suffix
            to_pil(preds[:3]).save(os.path.join(out_folder, f"{image_id}_albedo.png"))
            to_pil(preds[3]).save(os.path.join(out_folder, f"{image_id}_roughness.png"))
            to_pil(preds[4]).save(os.path.join(out_folder, f"{image_id}_metal.png"))
        else:
            # Save as dataset
            writeEXR(preds[:3].cpu().permute(1, 2, 0).numpy(), os.path.join(out_folder, "albedo", f"{image_id}.exr"))
            writeEXR(preds[3:].cpu().permute(1, 2, 0).numpy(), os.path.join(out_folder, "material", f"{image_id}.exr"))

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
            "input": Linear_2_SRGB()(img[0]),
            "albedo_pred": preds[:3],
            "roughness_pred": preds[3],
            "metal_pred": preds[4],
        })
        log_anything(logger, "stage2_material", logs)


def predict_materials(model, img, num_samples, sampling_batch_size=1, original_size=None):
    # Run model
    preds = []
    for _ in range(num_samples // sampling_batch_size):
        preds.append(
            model.sample(batch_size=sampling_batch_size,  # If more VRAM is available, can increase this number
                         conditioning_img=img.to(model.device)))
    assert len(preds) > 0, "No samples were generated"
    preds = torch.cat(preds, dim=0)

    # Resize the output to the original size
    if original_size is not None:
        preds = Resize(size=original_size)(preds)
    preds = preds.mean(0)

    return preds


@hydra.main(version_base="1.3", config_path="../../configs", config_name="stage/material_diffusion.yaml")
def main(cfg: DictConfig):
    material_diffusion(cfg)


if __name__ == "__main__":
    main()
