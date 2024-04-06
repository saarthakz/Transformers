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
        decay=0,
        **kwargs,
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
        self.quantize = (
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
