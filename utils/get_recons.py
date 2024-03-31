import torch
from torch import nn
from torchvision.utils import save_image
import os
from random import random


def get_recons(
    model: nn.Module,
    model_name: str,
    x: torch.Tensor,
    mean: float,
    std: float,
    with_vq=False,
):

    imgs = x.mul(std).add(mean)
    if with_vq:
        y, indices, loss = model.forward(x)
    else:
        y = model.forward(x)

    recon = y.mul(std).add(mean)
    base_path = os.path.join("images", model_name, "recons")
    os.makedirs(base_path, exist_ok=True)

    rand = str(int((random() * 100)))

    image_fp = os.path.join(base_path, f"{rand}_images.png")
    recon_fp = os.path.join(base_path, f"{rand}_recons.png")

    save_image(imgs, fp=image_fp)
    save_image(recon, fp=recon_fp)
