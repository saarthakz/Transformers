import os
import sys
import math
import torch
from torch import nn
from torch.nn import functional

sys.path.append(os.path.abspath("."))
from classes.Transformers import Block, FeedForward, MultiHeadAttention
from classes.SpectralNorm import SpectralNorm


class OverlappingPatchEmbedding(nn.Module):

    def __init__(
        self, input_res: int, num_channels: int, embed_dim: int, patch_size: int
    ):
        super().__init__()
        
        assert patch_size % 4 == 0, "Patch size must be a multiple of 4"

        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.input_res = input_res

        self.merger = nn.Unfold(
            kernel_size=self.patch_size, 
            stride=self.patch_size // 2, 
            padding=self.patch_size // 4,
        )
        self.proj = nn.Linear(
            in_features=(patch_size**2) * num_channels, out_features=embed_dim
        )

    def forward(self, x: torch.Tensor):
        x = self.merger.forward(x)
        x = x.transpose(-2, -1)
        x = self.proj.forward(x)
        return x


class OverlappingPatchUnembedding(nn.Module):
    def __init__(
        self, image_size: int, num_channels: int, embed_dim: int, patch_size: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.image_size = image_size

        self.proj = nn.Linear(
            in_features=embed_dim, out_features=(patch_size**2) * num_channels
        )
        self.merger = nn.Fold(
            output_size=image_size,
            kernel_size=patch_size,
            stride=patch_size // 2,
            padding=patch_size // 4,
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = x.transpose(-2, -1)
        x = self.merger(x)
        return x


class PatchEmbeddings(nn.Module):
    """
    Convert the image into non overlapping patches and then project them into a vector space.
    """

    def __init__(self, num_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(
            self.num_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor):
        # (batch_size, num_channels, input_res, input_res) -> (batch_size, num_patches, hidden_size)
        x = self.projection.forward(x)
        x = x.flatten(-2).transpose(-2, -1)
        return x


# Extrapolation of the PatchMerger by Google
class PatchModifier(nn.Module):
    def __init__(self, dim: int, num_tokens_out: int, use_scale=True):
        super().__init__()
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = dim**-0.5
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        sim = self.queries @ x.transpose(1, 2)
        if self.use_scale:
            sim = sim * self.scale
        attn = functional.softmax(sim, dim=-1)
        return attn @ x


class PoolDownsample(nn.Module):
    def __init__(self, input_res: list[int], kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.input_res = input_res
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        H, W = self.input_res
        assert L == H * W, "Resolution mismatch"
        x = x.transpose(-2, -1).view(B, C, H, W)
        max_pool_features = self.max_pool.forward(x)
        avg_pool_features = self.avg_pool.forward(x)
        features = torch.cat([max_pool_features, avg_pool_features], dim=1)
        features = features.view(B, 2 * C, -1).transpose(-2, -1)
        return features


class Upsample(nn.Module):
    def __init__(self, input_res: list[int], dim: int):
        super().__init__()
        self.input_res = input_res
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.proj = nn.Linear(in_features=dim, out_features=dim // 2)

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        H, W = self.input_res
        assert L == H * W, "Resolution mismatch"
        x = x.transpose(-2, -1).view(B, C, H, W)
        x = self.upsample.forward(x)
        x = x.view(B, C, -1).transpose(-2, -1)
        x = self.proj.forward(x)
        return x


class Upscale(nn.Module):
    def __init__(self, num_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.upscale = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=num_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor, height: int, width: int):
        B, C, D = x.shape

        assert (
            height * width == C
        ), f"Image dimensions don't match; {height} * {width} != {C} "
        x = x.transpose(1, 2)  # B, D, C
        x = x.view(B, D, height, width)
        img = self.upscale.forward(input=x)
        return img


class ViTEncoder(nn.Module):
    def __init__(
        self,
        input_res: list[int],
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
    ) -> None:
        super().__init__()
        self.input_res = input_res
        self.patch_size = patch_size

        H, W = self.input_res

        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.patch_embedding = PatchEmbeddings(num_channels, embed_dim, patch_size)

        scale = embed_dim**-0.5
        num_patches = H * W // (patch_size**2)
        patch_res = H // patch_size, W // patch_size
        self.patch_res = patch_res
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.pre_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(
            *[Block(embed_dim, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.initialize_weights()

    # Initializes all the layer weight with a normalized value
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
        num_heads: int,
        num_blocks: int,
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
        num_patches = H * W // (patch_size**2)
        patch_res = H // patch_size, W // patch_size
        self.patch_res = patch_res

        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.post_net_norm = nn.LayerNorm(embed_dim)

        self.transformer = nn.Sequential(
            *[Block(embed_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.proj = Upscale(num_channels, embed_dim, patch_size)

        self.initialize_weights()

    def forward(self, x):
        H, W = self.input_res
        x = x + self.position_embedding
        x = self.transformer(x)
        x = self.post_net_norm(x)
        x = self.proj(x, *self.patch_res)

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
