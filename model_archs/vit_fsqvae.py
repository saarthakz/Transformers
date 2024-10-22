from typing import List
from classes.Swin import res_scaler
from classes.VIT import (
    ViTEncoder,
    ViTDecoder,
)


from classes.FSQ import FSQ
import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))


class Model(nn.Module):
    """
    Args:
        input_res (List[int]): Image size as (H, W),
        patch_size (int): Patch size,
        num_channels (int): Number of input image channels,
        embed_dim (int): Initial Patch embedding dimension,
        num_codebook_embeddings (int): Number of codebook embeddings,
        codebook_dim (int): Codebook embedding dimension,
        num_heads (int): Number of heads to be used in attention blocks,
        num_blocks (int): Number of blocks in both encoder and decoder,
        dropout (int): Attention Dropout,
        beta (int): Commitment factor for calculation of codebook loss,
        decay (float): Decay for VectorQuantizer with EMA training
    """

    def __init__(
        self,
        input_res: List[int],
        dim: int,
        patch_size: int,
        num_channels: int,
        codebook_levels: List[int],
        num_heads: List[int],
        dropout: int,
        **kwargs,
    ):
        super().__init__()

        self.patch_res = res_scaler(input_res, 1 / patch_size)

        self.encoder = ViTEncoder(
            input_res,
            patch_size,
            num_channels,
            dim,
            num_heads,
            dropout,
        )
        self.decoder = ViTDecoder(
            input_res,
            patch_size,
            num_channels,
            dim,
            num_heads,
            dropout,
        )
        self.quantizer = FSQ(levels=codebook_levels, dim=dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        x = x.clamp(-1.0, 1.0)
        return x

    def quantize(self, x_enc: torch.Tensor):
        return self.quantizer.forward(x_enc)

    def forward(self, x: torch.Tensor):
        x_enc = self.encode(x)  # Encoder
        z_q, indices = self.quantize(x_enc)  # Scalar Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        loss = torch.tensor(data=0)  # To mimic the VQ loss which does not exists in FSQ
        return recon_imgs, indices, loss

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs
