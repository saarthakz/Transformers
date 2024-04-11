import os
import sys
import math
import torch
from torch import nn
from torch.nn import functional

sys.path.append(os.path.abspath("."))
from classes.Transformers import Block, FeedForward, MultiHeadAttention
from classes.SpectralNorm import SpectralNorm
from classes.Swin import MultiSwinBlock


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


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels=3, dim=128, patch_size=4) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.dim = (dim,)
        self.patch_size = patch_size

        self.patcher = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.lin = nn.Linear(
            in_features=num_channels * (patch_size**2), out_features=dim
        )

    def forward(self, x: torch.Tensor):
        x = self.patcher.forward(x)
        x = x.transpose(-1, -2)
        x = self.lin.forward(x)
        return x


class PatchUnembedding(nn.Module):
    def __init__(
        self, input_res: list[int], num_channels=3, dim=128, patch_size=4
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.dim = (dim,)
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
        num_layers: int,
        dropout: int,
        with_swin: bool,
        swin_depth: int,
        window_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_res = input_res
        self.patch_size = patch_size

        H, W = self.input_res

        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.patch_embedding = PatchEmbedding(num_channels, embed_dim, patch_size)

        scale = embed_dim**-0.5
        num_patches = H * W // (patch_size**2)
        patch_res = H // patch_size, W // patch_size
        self.patch_res = patch_res
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.pre_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.ModuleList()
        for _ in range(num_layers):
            if with_swin:
                self.transformer.append(
                    MultiSwinBlock(
                        embed_dim, patch_res, swin_depth, num_heads, window_size
                    )
                )
            for _ in range(num_blocks):
                self.transformer.append(Block(embed_dim, num_heads, dropout))

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
        for layer in self.transformer:
            x = layer.forward(x)

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
        num_layers: int,
        dropout: int,
        with_swin: bool,
        swin_depth: int,
        window_size: int,
        **kwargs,
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

        self.transformer = nn.ModuleList()
        for _ in range(num_layers):
            if with_swin:
                self.transformer.append(
                    MultiSwinBlock(
                        embed_dim, patch_res, swin_depth, num_heads, window_size
                    )
                )
            for _ in range(num_blocks):
                self.transformer.append(Block(embed_dim, num_heads, dropout))

        self.proj = Upscale(num_channels, embed_dim, patch_size)

        self.initialize_weights()

    def forward(self, x):
        H, W = self.input_res
        x = x + self.position_embedding
        for layer in self.transformer:
            x = layer.forward(x)
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
