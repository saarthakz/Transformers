import torch
from torch import nn
from einops.layers.torch import Rearrange


class PatchShiftAugmentation(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.shift = patch_size // 2
        self.pad = nn.ConstantPad2d(padding=self.shift, value=0)

    def forward(self, x: torch.Tensor):
        x_pad = self.pad.forward(x)

        x_up_pad = x_pad[..., : -self.shift * 2, self.shift : -self.shift]
        x_left_pad = x_pad[..., self.shift : -self.shift, : -self.shift * 2]

        return torch.cat([x, x_up_pad, x_left_pad], dim=-3)


class ShiftedPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_channels,
        embed_dim,
        patch_size=2,
    ):
        super().__init__()

        self.patch_shift_augmentation = PatchShiftAugmentation(patch_size)

        patch_dim = (num_channels * 3) * (patch_size**2)

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

        out = self.patch_shift_augmentation(x)
        out = self.patch_embedding(out)

        return out
