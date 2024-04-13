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
from einops import rearrange, einsum


def stable_softmax(t, dim=-1, alpha=32**2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


class LayerNormChan(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class ConvAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = LayerNormChan(dim)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()

        x = self.pre_norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=h), (q, k, v)
        )

        sim = einsum(q, k, "b h c i, b h c j -> b h i j") * self.scale

        attn = stable_softmax(sim, dim=-1)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b h i j, b h c j -> b h c i")
        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=height, y=width)
        out = self.to_out(out)

        return out + residual


class ConvPatchEmbedding(nn.Module):
    """
    Convert the image into non overlapping patches and then project them into a vector space.
    """

    def __init__(self, num_channels: int, dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = dim

        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(
            self.num_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection.forward(x)
        x = x.flatten(-2).transpose(-2, -1)
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


class ConvPatchUnembedding(nn.Module):
    def __init__(
        self,
        patch_res: list[int],
        num_channels: int,
        dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_res = patch_res
        self.patch_size = patch_size
        self.upscale = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=num_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor):
        B, C, D = x.shape
        H, W = self.patch_res

        assert H * W == C, f"Image dimensions don't match; {H} * {W} != {C} "
        x = x.transpose(1, 2)  # B, D, C
        x = x.view(B, D, H, W)
        img = self.upscale.forward(input=x)
        return img


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
