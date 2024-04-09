import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from classes.VectorQuantizer import VectorQuantizer, VectorQuantizerEMA
from classes.VIT import (
    ViTEncoder,
    ViTDecoder,
)


class ViTVQVAE(nn.Module):
    """
    Args:
        input_res (list[int]): Image size as (H, W),
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
        input_res: list[int],
        patch_size: int,
        num_channels: int,
        num_codebook_embeddings: int,
        codebook_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: int,
        beta: int,
        decay=0,
        **kwargs,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            input_res,
            patch_size,
            num_channels,
            embed_dim,
            num_heads,
            num_blocks,
            dropout,
        )
        self.decoder = ViTDecoder(
            input_res,
            patch_size,
            num_channels,
            embed_dim,
            num_heads,
            num_blocks,
            dropout,
        )
        self.quantizer = (
            VectorQuantizer(num_codebook_embeddings, codebook_dim, beta)
            if decay == 0
            else VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, beta, decay)
        )
        self.pre_quant = nn.Linear(embed_dim, codebook_dim)
        self.post_quant = nn.Linear(codebook_dim, embed_dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.pre_quant(x)
        return x

    def decode(self, x: torch.Tensor):
        x = self.post_quant(x)
        x = self.decoder(x)
        # x = x.clamp(-1.0, 1.0)
        return x

    def quantize(self, x_enc: torch.Tensor):
        B, C, D = x_enc.shape
        patch_H, patch_W = self.encoder.patch_res

        assert (
            C == patch_H * patch_W
        ), "Input patch length does not match the patch resolution"

        x_enc = x_enc.transpose(-2, -1).view(B, D, patch_H, patch_W)
        z_q, indices, loss = self.quantizer.forward(x_enc)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
        return z_q, indices, loss

    def forward(self, x: torch.Tensor):
        x_enc = self.encode(x)  # Encoder
        z_q, indices, loss = self.quantize(x_enc)  # Vector Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        return recon_imgs, indices, loss

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs


class SwinViTVQVAE(nn.Module):
    """
    Args:
        input_res (list[int]): Image size as (H, W),
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
        input_res: list[int],
        dim: int,
        patch_size: int,
        num_channels: int,
        num_codebook_embeddings: int,
        codebook_dim: int,
        num_heads: int,
        num_blocks: int,
        num_layers: int,
        dropout: int,
        beta: int,
        decay=0.01,
        with_swin=False,
        swin_depth=2,
        window_size=4,
        **kwargs,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            input_res,
            patch_size,
            num_channels,
            dim,
            num_heads,
            num_blocks,
            num_layers,
            dropout,
            with_swin,
            swin_depth,
            window_size,
        )
        self.decoder = ViTDecoder(
            input_res,
            patch_size,
            num_channels,
            dim,
            num_heads,
            num_blocks,
            num_layers,
            dropout,
            with_swin,
            swin_depth,
            window_size,
        )
        self.quantizer = (
            VectorQuantizer(num_codebook_embeddings, codebook_dim, beta)
            if decay == 0
            else VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, beta, decay)
        )
        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.post_quant = nn.Linear(codebook_dim, dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.pre_quant(x)
        return x

    def decode(self, x: torch.Tensor):
        x = self.post_quant(x)
        x = self.decoder(x)
        # x = x.clamp(-1.0, 1.0)
        return x

    def quantize(self, x_enc: torch.Tensor):
        B, C, D = x_enc.shape
        patch_H, patch_W = self.encoder.patch_res

        assert (
            C == patch_H * patch_W
        ), "Input patch length does not match the patch resolution"

        x_enc = x_enc.transpose(-2, -1).view(B, D, patch_H, patch_W)
        z_q, indices, loss = self.quantizer.forward(x_enc)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
        return z_q, indices, loss

    def forward(self, x: torch.Tensor):
        x_enc = self.encode(x)  # Encoder
        z_q, indices, loss = self.quantize(x_enc)  # Vector Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        return recon_imgs, indices, loss

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs
