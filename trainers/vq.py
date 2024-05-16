import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
from utils.logger import Logger
from utils.get_recons import get_recons
from utils.get_dataset import get_dataset
from utils.get_model_arch import get_model_arch
from utils.get_optimizer import get_optimizer
import math
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
import json
import wandb
from datetime import datetime


def main(config: dict):
    model_name = (
        f'{config["model_name"]} {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'
    )
    model_dir = os.path.join(os.getcwd(), "models", model_name)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config["acc_find_unused_params"]
    )
    accelerator = Accelerator(
        project_dir=model_dir,
        log_with="wandb",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    # Print the config file
    accelerator.print(config)

    if config["tracking"]:
        accelerator.init_trackers(
            project_name="Major-Project",
            config=config,
            init_kwargs={
                "wandb": {
                    "name": model_name,
                    "entity": "tangentmay",
                },
            },
        )

    if accelerator.is_main_process:
        os.makedirs(name=model_dir, exist_ok=True)
        logger = Logger(os.path.join(model_dir, "log.txt"))

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Dataset and Dataloaders

    train_dataset = get_dataset(
        config["dataset"],
        config["input_res"],
        config["dataset_mean"],
        config["dataset_std"],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Model
    model = get_model_arch(config["model_arch"])(**config)
    accelerator.print(model)

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

    # Optimizers
    optim = get_optimizer(config["optimizer"])(model.parameters(), lr=config["lr"])

    # Acceleration :P
    train_loader = accelerator.prepare_data_loader(data_loader=train_loader)
    model = accelerator.prepare_model(model=model)
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
                optim.zero_grad()
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
                            "Images": wandb.Image(
                                images, caption=f"Step {total_steps}"
                            ),
                            "Recons": wandb.Image(
                                recons, caption=f"Step {total_steps}"
                            ),
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
