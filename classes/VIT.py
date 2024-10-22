import os
import sys
import math
import torch
from torch import nn
from torch.nn import functional

sys.path.append(os.path.abspath("."))
from classes.Transformers import Block
from classes.Swin import res_scaler
from einops import rearrange, einsum


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Linear layer to project flattened patches to embedding dimension
        self.projection = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            "Image dimensions must be divisible by the patch size."

        # Rearrange the image to patches: [batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size]
        # Then flatten the patches: [batch_size, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        
        # Project the patches to the embedding dimension
        x = self.projection(x)  # [batch_size, num_patches, embed_dim]
        return x


class PatchToImage(nn.Module):
    def __init__(
        self, input_res: list[int], num_channels=3, dim=128, patch_size=4
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.dim = dim
        self.patch_size = patch_size
        self.input_res = input_res

        self.lin = nn.Linear(
            in_features=dim, out_features=num_channels * (patch_size**2)
        )
        self.patcher = nn.Fold(
            output_size=input_res, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        x = self.lin.forward(x)
        x = x.transpose(-1, -2)
        x = self.patcher.forward(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        input_res: list[int],
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_heads: list[int],
        dropout: int,
    ) -> None:
        super().__init__()
        self.input_res = input_res
        self.patch_size = patch_size
        H, W = self.input_res

        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.patch_embedding = PatchEmbedding(num_channels, patch_size, embed_dim)

        scale = embed_dim**-0.5
        self.patch_res = res_scaler(self.input_res, 1 / self.patch_size)
        patch_H, patch_W = self.patch_res
        num_patches = patch_H * patch_W
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.pre_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(
            *[Block(embed_dim, heads, dropout) for heads in num_heads]
        )

        self.initialize_weights()

    # Initializes all the layer weightd with a normalized value
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding.forward(x)
        x = x + self.position_embedding
        x = self.pre_net_norm(x)
        x = self.transformer(x)

        return x


class ViTDecoder(nn.Module):
    def __init__(
        self,
        input_res: list[int],
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_heads: list[int],
        dropout: int,
    ) -> None:
        super().__init__()

        self.input_res = input_res
        self.patch_size = patch_size
        H, W = self.input_res

        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        scale = embed_dim**-0.5
        self.patch_res = res_scaler(self.input_res, 1 / self.patch_size)
        patch_H, patch_W = self.patch_res
        num_patches = patch_H * patch_W
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.post_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(
            *[Block(embed_dim, heads, dropout) for heads in num_heads]
        )
        self.proj = PatchToImage(
            input_res, num_channels, embed_dim, patch_size
        )

        self.initialize_weights()

    def forward(self, x):
        x = x + self.position_embedding
        x = self.transformer(x)
        x = self.post_net_norm(x)
        x = self.proj(x)

        return x

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
