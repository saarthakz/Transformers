import torch
import torch.nn as nn
import os
import sys
from timm.models.layers import trunc_normal_

sys.path.append(os.path.abspath("."))
from classes.VIT import (
    PatchEmbedding,
    PatchUnembedding,
    Block,
)
from classes.VectorQuantizer import VectorQuantizerEMA
from classes.StyleSwin import PoolDownsample, BilinearUpsample
from classes.SPT import ShiftedPatchEmbedding
from classes.Swin import res_scaler


class Model(nn.Module):
    def __init__(
        self,
        dim=192,
        input_res: list[int] = [64, 64],
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        num_blocks=[2, 2],
        num_heads=[3, 6],
        with_shifted_patch_embeddings=False,
        dropout=0.01,
        **kwargs
    ):
        super().__init__()

        self.with_shifted_patch_embeddings = with_shifted_patch_embeddings
        self.num_layers = len(num_heads)
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.patch_embedding = (
            ShiftedPatchEmbedding(num_channels, dim, patch_size)
            if with_shifted_patch_embeddings
            else PatchEmbedding(num_channels, dim, patch_size)
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for idx in range(self.num_layers):
            for _ in range(self.num_blocks[idx]):
                self.encoder.append(Block(dim, num_heads[idx], dropout))

            self.encoder.append(PoolDownsample(res, dim, dim * 2))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.num_heads.reverse()

        # Decoder Layers
        for idx in range(self.num_layers):
            for _ in range(self.num_blocks[idx]):
                self.decoder.append(Block(dim, num_heads[idx], dropout))

            self.decoder.append(BilinearUpsample(res, dim, dim // 2))

            dim = dim // 2
            res = res_scaler(res, 2)

        self.patch_unembedding = PatchUnembedding(
            input_res, num_channels, dim, patch_size
        )
        self.apply(self._init_weights)

    def encode(self, x: torch.Tensor):
        x = self.patch_embedding.forward(x)
        for layer in self.encoder:
            x = layer.forward(x)
        return x

    def decode(self, z_q: torch.Tensor):
        for layer in self.decoder:
            z_q = layer.forward(z_q)

        z_q = self.patch_unembedding.forward(z_q)

        return z_q

    def forward(self, img: torch.Tensor):
        x_enc = self.encode(img)  # Encoder
        recon_imgs = self.decode(x_enc)  # Decoder
        return recon_imgs, None, 0

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs

    def _init_weights(self, m):
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
