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
from model_classes.VITVQVAE import SwinViTVQVAE
import wandb


def main(config: dict):
    model_name = config["model_name"]
    model_dir = os.path.join(os.getcwd(), "models", model_name)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=model_dir,
        log_with="wandb",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    # Print the config file
    accelerator.print(config)

    accelerator.init_trackers(
        project_name="Major-Project",
        config=config,
        init_kwargs={
            "wandb": {"name": model_name, "entity": "tangentmay"},
        },
    )

    if accelerator.is_main_process:
        os.makedirs(name=model_dir, exist_ok=True)
        logger = Logger(os.path.join(model_dir, "log.txt"))

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Dataset and Dataloaders
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=(178, 178)),
            transforms.Resize(size=(64, 64), antialias=True),
            transforms.Normalize(
                mean=[0.5084, 0.4224, 0.3767],
                std=[0.3012, 0.2788, 0.2773],
            ),
        ]
    )

    train_dataset = datasets.CelebA(
        root="./data",
        split="train",
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    train_loader = accelerator.prepare_data_loader(data_loader=train_loader)

    # Model
    model = SwinViTVQVAE(**config)

    # Print # of model parameters
    accelerator.print(
        sum(param.numel() for param in model.parameters()) / 1e6, "M parameters"
    )

    # Load a model checkpoint
    if config["model_from_checkpoint"]:
        model.load_state_dict(torch.load(f=config["model_checkpoint_path"]))
        accelerator.print(
            "Model loaded from checkpoint: ", config["model_checkpoint_path"]
        )

    model = accelerator.prepare_model(model=model)

    # Optimizers
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    optim = accelerator.prepare_optimizer(optimizer=optim)

    # Load a state from checkpoint if required
    if config["state_from_checkpoint"]:
        accelerator.load_state(input_dir=config["state_checkpoint_path"])
        accelerator.print(
            "State loaded from checkpoint: ", config["state_checkpoint_path"]
        )

    total_steps = epochs * len(train_loader)
    checkpoint_step = total_steps // config["num_checkpoints"]
    accelerator.print(
        f"Total steps: {total_steps} and checkpoint every {checkpoint_step} steps"
    )

    if accelerator.is_main_process:
        progress_bar = tqdm(range(epochs * len(train_loader)))

    total_steps = 0
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                x, y = batch

                # evaluate the loss
                recon, codebook_indices, q_loss = model.forward(x)
                recon_loss = nn.functional.mse_loss(x, recon)
                loss: torch.Tensor = q_loss + recon_loss
                epoch_loss += loss.detach().item()
                optim.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optim.step()
                total_steps += 1

                accelerator.log({"train_loss": loss}, step=total_steps)

                if total_steps % checkpoint_step == 0:
                    ckpt_dir = os.path.join(model_dir, "checkpoints", f"{total_steps}")
                    images, recons = get_recons(
                        model=model,
                        dir=ckpt_dir,
                        x=x[: config["recon_batch_size"]],
                        with_vq=config["with_vq"],
                        return_image=True,
                    )
                    accelerator.log(
                        {
                            "wandb": {
                                "Images": wandb.Image(
                                    images, caption=f"Step {total_steps}"
                                ),
                                "Recons": wandb.Image(
                                    recons, caption=f"Step {total_steps}"
                                ),
                            }
                        }
                    )
                    accelerator.save_state(
                        ckpt_dir,
                        safe_serialization=False,
                    )

                if accelerator.is_main_process:
                    progress_bar.update(1)

        total_loss += epoch_loss
        if accelerator.is_main_process:
            epoch_loss_log = f"Epoch: {epoch}, Avg Epoch Loss {epoch_loss / (step + 1)}, Net Avg Loss: {total_loss / (total_steps + 1)}"
            logger.log(epoch_loss_log)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.save_model(model, os.path.join(model_dir), safe_serialization=False)


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
