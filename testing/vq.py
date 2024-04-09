import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
from utils.logger import Logger
from utils.get_recons import get_recons
import math
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
import json
from model_classes.VIT_PoolDownsample_BilinearUpsample import (
    ViT_PoolDownsample_BilinearUpsample,
    ViT_SS_PoolDown_BilinUp,
)
from model_classes.VIT_PatchModifier import (
    ViT_PatchMergeExpand,
    ViT_OverlapPatchMergeExpand,
)


def main(config: dict):
    model_name = config["model_name"]
    model_dir = os.path.join(os.getcwd(), "models", model_name)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=model_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # Print the config file
    accelerator.print(config)

    # Dataset and Dataloaders
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=(178, 178)),
            transforms.Resize(size=(64, 64), antialias=True),
        ]
    )

    test_dataset = datasets.CelebA(
        root="./data",
        split="test",
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["recon_batch_size"],
        shuffle=True,
    )

    test_loader = accelerator.prepare_data_loader(data_loader=test_loader)

    # Model
    model = ViT_SS_PoolDown_BilinUp(**config)

    # Print # of model parameters
    accelerator.print(
        sum(param.numel() for param in model.parameters()) / 1e6, "M parameters"
    )

    # Load model checkpoint
    if config["model_from_checkpoint"]:
        model.load_state_dict(torch.load(f=config["model_checkpoint_path"]))
        accelerator.print(
            "Model loaded from checkpoint: ", config["model_checkpoint_path"]
        )

    model = accelerator.prepare_model(model=model)

    x, y = next(iter(test_loader))
    get_recons(
        model=model,
        model_name=config["model_name"],
        x=x,
        with_vq=config["with_vq"],
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config JSON file",
    )
    args = vars(arg_parser.parse_args())
    config_file = open(file=args["config_file"], mode="r")
    config = json.load(config_file)
    main(config)
