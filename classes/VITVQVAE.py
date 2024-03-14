import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from classes.VectorQuantizer import VectorQuantizer
from classes.VIT import (
    ViTEncoder,
    ViTDecoder,
    PatchEmbeddings,
    PatchModifier,
    Upscale,
    Block,
)


class ViTVQVAE(nn.Module):
    """
    Args:
        image_size (tuple[int]): Image size as (H, W),
        patch_size (int): Patch size,
        num_channels (int): Number of input image channels,
        embed_dim (int): Initial Patch embedding dimension,
        num_codebook_embeddings (int): Number of codebook embeddings,
        codebook_dim (int): Codebook embedding dimension,
        num_heads (int): Number of heads to be used in attention blocks,
        num_blocks (int): Number of blocks in both encoder and decoder,
        dropout (int): Attention Dropout,
        beta (int): Commitment factor for calculation of codebook loss,

    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        num_codebook_embeddings: int,
        codebook_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
        beta: int,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            image_size,
            patch_size,
            num_channels,
            embed_dim,
            num_heads,
            num_blocks,
            dropout,
        )
        self.decoder = ViTDecoder(
            image_size,
            patch_size,
            num_channels,
            embed_dim,
            num_heads,
            num_blocks,
            dropout,
        )
        self.quantize = VectorQuantizer(num_codebook_embeddings, embed_dim, beta)
        self.pre_quant = nn.Linear(embed_dim, codebook_dim)
        self.post_quant = nn.Linear(codebook_dim, embed_dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.encoder(x)
        x = self.pre_quant(x)
        return x

    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        # x = x.clamp(-1.0, 1.0)
        return x

    def forward(self, x):
        x_enc = self.encode(x)  # Encoder
        B, C, D = x_enc.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x_enc = x_enc.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.quantize.forward(x_enc)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
        recon_img: torch.Tensor = self.decode(z_q)  # Decoder
        return recon_img, indices, loss

    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))


class ViTVQVAE_v2(nn.Module):
    """
    Args:
        image_size (tuple[int]): Image size as (H, W),
        patch_size (int): Patch size,
        num_channels (int): Number of input image channels,
        embed_dim (int): Initial Patch embedding dimension,
        num_codebook_embeddings (int): Number of codebook embeddings,
        codebook_dim (int): Codebook embedding dimension,
        num_heads (int): Number of heads to be used in attention blocks,
        num_blocks (int): Number of blocks in both encoder and decoder,
        dropout (int): Attention Dropout,
        beta (int): Commitment factor for calculation of codebook loss,

    For num_blocks > 1, after each transformer block, a patch modification layer is used to compress and decompress the patches in the encoder and decoder
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        num_codebook_embeddings: int,
        codebook_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
        beta: int,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        H, W = image_size

        assert (
            H % patch_size == 0 and W & patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.patch_embedding = PatchEmbeddings(num_channels, embed_dim, patch_size)

        scale = embed_dim**-0.5
        H, W = image_size
        num_patches = H * W // (patch_size**2)
        self.position_embedding = nn.Parameter(
            torch.randn(size=(1, num_patches, embed_dim)) * scale
        )
        self.pre_encoder_norm = nn.LayerNorm(embed_dim)

        encoder_layers = []

        for _ in range(num_blocks):
            encoder_layers.append(Block(embed_dim, num_heads, dropout))
            encoder_layers.append(Block(embed_dim, num_heads, dropout))
            num_patches = num_patches * 2
            encoder_layers.append(
                PatchModifier(dim=embed_dim, num_tokens_out=num_patches)
            )
            encoder_layers.append(nn.Linear(embed_dim, embed_dim * 2))
            embed_dim = embed_dim * 2

        self.encoder = nn.Sequential(*encoder_layers)

        self.pre_quant = nn.Linear(embed_dim, codebook_dim)
        self.quantize = VectorQuantizer(num_codebook_embeddings, codebook_dim, beta)
        self.post_quant = nn.Linear(codebook_dim, embed_dim)

        decoder_layers = []

        for _ in range(num_blocks):
            decoder_layers.append(Block(embed_dim, num_heads, dropout))
            decoder_layers.append(Block(embed_dim, num_heads, dropout))
            num_patches = num_patches // 2
            decoder_layers.append(
                PatchModifier(dim=embed_dim, num_tokens_out=num_patches)
            )
            decoder_layers.append(nn.Linear(embed_dim, embed_dim // 2))
            embed_dim = embed_dim // 2

        self.decoder = nn.Sequential(*decoder_layers)

        self.upscale = Upscale(num_channels, embed_dim, patch_size)

        self.compression_factor = 2**num_blocks

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        x = self.encoder(x)
        x = self.pre_quant(x)
        return x

    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        # x = x.clamp(-1.0, 1.0)
        return x

    def forward(self, x):
        H, W = self.image_size
        H, W = H // self.compression_factor, W // self.compression_factor

        x_enc = self.encode(x)  # Encoder
        B, C, D = x_enc.shape
        x_enc = x_enc.transpose(-2, -1).view(B, D, H, W)

        z_q, indices, loss = self.quantize.forward(x_enc)  # Vector Quantizer

        z_q = z_q.view(B, D, C).transpose(-2, -1)
        recon_img: torch.Tensor = self.decode(z_q)  # Decoder
        return recon_img, indices, loss

    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
