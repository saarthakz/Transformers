import os
import sys
import math
import torch
from torch import nn
from torch.nn import functional

sys.path.append(os.path.abspath("."))
from classes.Transformers import Block, FeedForward, MultiHeadAttention
from classes.SpectralNorm import SpectralNorm
from classes.VQVAE import ConvAttention


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
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection.forward(x)
        x = x.flatten(-2).transpose(-2, -1)
        return x


class PatchModifier(nn.Module):
    def __init__(self, dim, num_tokens_out, use_scale=True):
        super().__init__()
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = dim**-0.5
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = self.queries @ x.transpose(1, 2)
        if self.use_scale:
            sim = sim * self.scale
        attn = functional.softmax(sim, dim=-1)
        return attn @ x


class PoolDownsample(nn.Module):
    def __init__(self, input_res: tuple[int], kernel_size=3, stride=2, padding=1):
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
    def __init__(self, input_res: tuple[int], dim: int):
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


class VisionTransformerForClassification(nn.Module):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_classes: int,
        NUM_BLOCKS=3,
        dropout=0,
        device="cuda",
    ):
        super().__init__()

        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # Num patches == Context
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.mask = (
            torch.zeros(size=(self.num_patches + 1, self.num_patches + 1))
            .bool()
            .to(device=device)
        )

        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, num_channels, embed_dim
        ).to(device=device)
        self.cls_token = nn.Parameter(
            data=torch.randn(size=(1, 1, embed_dim), device=device)
        )
        self.position_embedding_table = nn.Embedding(
            self.num_patches + 1, embed_dim, device=device
        )
        self.blocks = nn.ModuleList(
            [
                Block(emb_dims=embed_dim, num_heads=4, dropout=dropout)
                for _ in range(NUM_BLOCKS)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Language model head used for output
        self.lm_head = nn.Linear(embed_dim, self.num_classes)

    def forward(self, x, targets=None):
        B, C, H, W = x.shape

        # x and targets are both (B,C) tensor of integers

        # Getting the Patch embeddings
        patch_emb: torch.Tensor = self.patch_embeddings(x)  # (B,C,D)
        cls_token = self.cls_token.expand(B, -1, -1)

        # Added the Class token to the Patch embeddings
        # (B, C+1, D) Added Class token
        x = torch.concat([cls_token, patch_emb], dim=1)

        B, C, D = x.shape

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(
            torch.arange(C, device=self.device)
        )  # (C,D)

        # Adding the position embedding to the patch embeddings
        x = x + pos_emb

        for block in self.blocks:
            x = block(x, self.mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        cls_logits = logits[:, 0]

        if targets is None:
            loss = None
        else:
            loss = functional.cross_entropy(cls_logits, targets)

        return cls_logits, loss

    def predict(self, x):
        # Get the predictions
        cls_logits, loss = self.forward(x)
        probs = functional.softmax(cls_logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        return predictions


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int],
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        H, W = self.image_size

        assert (
            H % patch_size == 0 and W % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.patch_embedding = PatchEmbeddings(num_channels, embed_dim, patch_size)

        scale = embed_dim**-0.5
        num_patches = H * W // (patch_size**2)
        patch_res = H // patch_size, W // patch_size
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
        image_size: int,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        H, W = self.image_size

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
        H, W = self.image_size
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
