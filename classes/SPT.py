import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math


class PatchShiftAugmentation(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1 / 2))
        self.pad = nn.ConstantPad2d(padding=self.shift, value=0)

    def forward(self, x: torch.Tensor):
        x_pad = self.pad.forward(x)

        x_lu = x_pad[..., : -self.shift * 2, : -self.shift * 2]
        x_ru = x_pad[..., : -self.shift * 2, self.shift * 2 :]
        x_lb = x_pad[..., self.shift * 2 :, : -self.shift * 2]
        x_rb = x_pad[..., self.shift * 2 :, self.shift * 2 :]

        return torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=-3)


class ShiftedPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_channels,
        embed_dim,
        patch_size=2,
        is_pe=False,
    ):
        super().__init__()

        self.patch_shift_augmentation = PatchShiftAugmentation(patch_size)

        patch_dim = (num_channels * 5) * (patch_size**2)

        self.is_pe = is_pe

        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
        )

    def forward(self, x):

        out = (
            x
            if self.is_pe
            else rearrange(x, "b (h w) d -> b d h w", h=int(math.sqrt(x.size(1))))
        )

        out = self.patch_shift_augmentation(out)
        out = self.patch_embedding(out)

        return out
