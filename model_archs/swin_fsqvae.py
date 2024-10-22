import os
import sys
from typing import List

sys.path.append(os.path.abspath("."))

from classes.Swin import res_scaler, MultiSwinBlock, ConvPatchMerge, ConvPatchExpand
from classes.VIT import (
    PatchEmbedding,
    PatchToImage,
    Block,
)
from classes.FSQ import FSQ
import torch
import torch.nn as nn


from timm.models.layers import trunc_normal_
import math


class Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_res: list[int] = [64, 64],
        patch_size=4,
        window_size=4,
        num_channels=3,
        codebook_levels: List[int] = [7, 5, 5, 5],
        swin_depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [4, 8, 8, 16],
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(num_heads)
        self.num_heads = num_heads
        self.swin_depths = swin_depths
        self.window_size = window_size

        assert len(num_heads) == len(
            swin_depths
        ), "num_heads and swin_depths must be the same length"

        self.patch_embedding = PatchEmbedding(
            num_channels,
            patch_size,
            dim,
        )
        self.patch_to_image = PatchToImage(
            input_res,
            num_channels,
            dim,
            patch_size,
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for idx in range(self.num_layers):
            self.encoder.append(
                MultiSwinBlock(
                    dim,
                    res,
                    swin_depths[idx],
                    num_heads[idx],
                    window_size=window_size,
                    final_layer=ConvPatchMerge(res, dim, dim * 2),
                )
            )
            res = res_scaler(res, 1 / 2)
            dim *= 2

        self.quantizer = FSQ(levels=codebook_levels, dim=dim)

        # Decoder Layers
        for idx in range(self.num_layers):
            self.decoder.append(
                MultiSwinBlock(
                    dim,
                    res,
                    swin_depths[idx],
                    num_heads[idx],
                    window_size=window_size,
                    final_layer=ConvPatchExpand(res, dim, dim // 2),
                )
            )
            res = res_scaler(res, 2)
            dim //= 2

        self.apply(self.init_weights)

    def encode(self, x: torch.Tensor):
        x = self.patch_embedding.forward(x)
        for layer in self.encoder:
            x = layer.forward(x)
        return x

    def decode(self, z_q: torch.Tensor):
        for layer in self.decoder:
            z_q = layer.forward(z_q)
        z_q = self.patch_to_image.forward(z_q)
        return z_q

    def quantize(self, x_enc: torch.Tensor):
        return self.quantizer.forward(x_enc)  # Vector Quantizer

    def forward(self, img: torch.Tensor):
        x_enc = self.encode(img)  # Encoder
        z_q, indices = self.quantize(x_enc)  # Scalar Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        loss = torch.tensor(data=0)  # To mimic the VQ loss which does not exists in FSQ
        return recon_imgs, indices, loss

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
