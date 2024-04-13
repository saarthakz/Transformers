import torch
import torch.nn as nn
import os
import sys
from timm.models.layers import trunc_normal_

sys.path.append(os.path.abspath("."))
from classes.VIT import PatchEmbedding, PatchUnembedding, Block, ConvAttention
from classes.VectorQuantizer import VectorQuantizerEMA, VectorQuantizer
from classes.Swin import res_scaler, ConvPatchExpand, ConvPatchMerge


class Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_res: list[int] = [64, 64],
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        num_blocks=[2, 2],
        num_heads=[3, 6],
        dropout=0.01,
        beta=0.5,
        decay=0.99,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(num_heads)
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.patch_embedding = nn.Conv2d(
            num_channels,
            dim,
            patch_size,
            patch_size,
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for idx in range(self.num_layers):
            for _ in range(self.num_blocks[idx]):
                self.encoder.append(
                    ConvAttention(
                        dim=dim,
                        dim_head=dim // num_heads[idx],
                        heads=num_heads[idx],
                        dropout=dropout,
                    )
                )

            self.encoder.append(ConvPatchMerge(dim, dim * 2))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = (
            VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, beta, decay)
            if int(decay)
            else VectorQuantizer(num_codebook_embeddings, codebook_dim, beta)
        )
        self.post_quant = nn.Linear(codebook_dim, dim)

        # Decoder Layers
        for idx in range(self.num_layers):
            for _ in range(self.num_blocks[idx]):
                self.decoder.append(
                    ConvAttention(
                        dim=dim,
                        dim_head=dim // num_heads[idx],
                        heads=num_heads[idx],
                        dropout=dropout,
                    )
                )
            self.decoder.append(ConvPatchExpand(dim, dim // 2))
            dim = dim // 2
            res = res_scaler(res, 2)

        self.patch_unembedding = nn.ConvTranspose2d(
            dim,
            num_channels,
            patch_size,
            patch_size,
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

    def quantize(self, x_enc: torch.Tensor):
        z_q, indices, loss = self.vq.forward(x_enc)  # Vector Quantizer
        return z_q, indices, loss

    def forward(self, img: torch.Tensor):
        x_enc = self.encode(img)  # Encoder
        z_q, indices, loss = self.quantize(x_enc)  # Vector Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        return recon_imgs, indices, loss

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
