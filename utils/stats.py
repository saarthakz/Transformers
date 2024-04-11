import torch
from torch.utils.data import DataLoader


def get_mean(
    dataloader: DataLoader,
    device: torch.device = "cpu",
):
    mean = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)
    return mean


def get_std(
    input_res: list[int],
    mean: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device = "cpu",
):
    H, W = input_res
    var = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(dataloader.dataset) * H * W))
    return std
