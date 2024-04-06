import torch
import torch.nn as nn
from utils.logger import Logger
from utils.get_recons import get_recons
import math
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from accelerate import Accelerator
import os
import sys
import argparse
from model_classes.VIT_PoolDownsample_BilinearUpsample import (
    ViT_PoolDownsample_BilinearUpsample,
)


def main(config: dict):

    model_dir = os.path.join(os.getcwd(), "models", config.model_name)
    os.makedirs(name=model_dir, exist_ok=True)
    # logger = Logger(os.path.join(model_dir, "log.txt"))
    epochs = 100
    batch_size = 256

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=(178, 178)),
            transforms.Resize(size=(64, 64), antialias=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    model = ViT_PoolDownsample_BilinearUpsample()

    print(sum(param.numel() for param in model.parameters()) / 1e6, "M parameters")

    train_dataset = datasets.CelebA(
        root="./data", split="train", download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    optim = torch.optim.Adam(model.parameters())

    progress_bar = tqdm(range(epochs * len(train_loader)))

    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            x, y = batch

            # evaluate the loss
            recon, codebook_indices, q_loss = model.forward(x)
            recon_loss = nn.functional.mse_loss(x, recon)
            loss: torch.Tensor = q_loss + recon_loss
            total_loss += loss.detach().item()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            progress_bar.update(1)

        train_loss_log = f"Epoch: {epoch}, Train loss {total_loss / (step + 1)}"
        logger.log(train_loss_log)

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config JSON file",
    )
    args = arg_parser.parse_args()
    main(args)
