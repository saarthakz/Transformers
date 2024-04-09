import torch
from torch import nn
from torchvision.utils import save_image, make_grid
import os
from random import random


def get_recons(
    model: nn.Module,
    dir: str,
    x: torch.Tensor,
    with_vq=False,
    nrow=4,
    return_image=False,
):

    if with_vq:
        y, indices, loss = model.forward(x)
    else:
        y = model.forward(x)

    base_path = os.path.join(dir, "recons")
    os.makedirs(base_path, exist_ok=True)

    rand = str(int((random() * 100)))

    image_fp = os.path.join(base_path, f"{rand}_images.png")
    recon_fp = os.path.join(base_path, f"{rand}_recons.png")

    images = make_grid(x, nrow=nrow)
    recons = make_grid(y, nrow=nrow)

    save_image(images, fp=image_fp)
    save_image(recons, fp=recon_fp)

    if return_image:
        return images, recons
