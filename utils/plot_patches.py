import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


def plot_patches(x: torch.Tensor, nrows_ncols=[8, 8]):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

    for idx, ax in enumerate(grid):
        patch = x[idx].permute(1, 2, 0).numpy()
        ax.imshow(patch)
        ax.axis("off")

    fig.show()
