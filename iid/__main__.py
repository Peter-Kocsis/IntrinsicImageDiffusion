import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
from argparse import ArgumentParser

from iid.iid import IntrinsicImageDiffusion
import omegaconf
import torch
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

from iid.transform import SRGB_2_Linear
from iid.utils import init_logger, readPNG


def __iid_argsparser():
    parser = ArgumentParser(description="Script for running the Intrinsic Image Diffusion model.",
                            add_help=True)
    # ============================== MODEL ==============================
    parser.add_argument(
        "-conf", "--config_path", type=str, default="models/config.yaml",
        help="Model config path"
    )
    parser.add_argument(
        "-ckpt", "--ckpt_path", type=str, default="models/iid_e250.pth",
        help="Model checkpoint path"
    )

    # ============================== DATA ==============================
    parser.add_argument(
        "-i", "--input", type=str, default="res/test.png",
        help="Input image path"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output/test_out.png",
        help="Input image path"
    )

    # ============================== SAMPLING ==============================
    parser.add_argument(
        "-ns", "--num_samples", type=int, default=10,
        help="Number of diffusion samples to average"
    )

    return parser


if __name__ == '__main__':
    argsparser = __iid_argsparser()
    args = argsparser.parse_args()

    config_path = args.config_path
    ckpt_path = args.ckpt_path

    logger = init_logger("MAIN")

    # ============= MODEL =============

    # Load config
    config = omegaconf.OmegaConf.load(config_path)

    # Init model
    model = IntrinsicImageDiffusion(unet_config=config.unet_config,
                                    diffusion_config=config.diffusion_config,
                                    ddim_config=config.ddim_config)

    # Load weights
    logger.info(f"Loading model <{ckpt_path}>")
    ckpt = torch.load(ckpt_path)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}")

    model.load_state_dict(ckpt)
    model = model.to(device)

    # ============= INFERENCE =============

    # Read input
    img_path = args.input
    img = readPNG(img_path)
    original_size = img.shape[:2][::-1]

    # Prepare batch
    transforms = Compose([ToTensor(), SRGB_2_Linear(), Resize(size=[480, 640])])
    img = transforms(img).unsqueeze(0).to(device)

    # Run model
    preds = []
    for _ in range(args.num_samples):
        preds.append(model.sample(batch_size=1,  # If more VRAM is available, can increase this number
                                  conditioning_img=img))
    preds = torch.cat(preds, dim=0)

    # Resize the output to the original size
    preds = Resize(size=original_size)(preds)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_filename = Path(args.output)
    pred = ToPILImage()(preds[:,:3].mean(0)).resize(original_size)
    pred.save(args.output)
    pred = ToPILImage()(preds[:,3].mean(0)).resize(original_size)
    pred.save(str(out_filename.with_name(f"{out_filename.stem}_roughness.png")))
    pred = ToPILImage()(preds[:,4].mean(0)).resize(original_size)
    pred.save(str(out_filename.with_name(f"{out_filename.stem}_metal.png")))



