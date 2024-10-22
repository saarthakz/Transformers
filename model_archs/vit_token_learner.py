from vector_quantize_pytorch import VectorQuantize
from classes.TokenLearner import TokenLearner, TokenToSpatialTransformer
from classes.SPT import ShiftedPatchEmbedding
from classes.Swin import res_scaler
from classes.VIT import (
    PatchEmbedding,
    PatchUnembedding,
    Block,
)
import torch
import torch.nn as nn
import os
import sys
from timm.models.layers import trunc_normal_
import math

sys.path.append(os.path.abspath("."))


class Model(nn.Module):
    def __init__(
        self,
        dim=128,
        input_res: list[int] = [64, 64],
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        num_tokens=16,
        num_blocks=[2, 2],
        num_heads=[4, 8],
        beta=0.25,
        decay=0.99,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(num_heads)
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        assert len(num_blocks) == len(
            num_heads
        ), "num_blocks and num_heads must be the same length"

        token_learner_insertion_idx = len(num_blocks) // 2

        self.patch_embedding = PatchEmbedding(num_channels, dim, patch_size)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for idx in range(token_learner_insertion_idx):
            self.encoder.extend(
                [Block(dim, num_heads[idx]) for _ in range(num_blocks[idx])]
            )

        # Add the TokenLearner at the half point
        self.encoder.append(TokenLearner(num_tokens, dim))

        for idx in range(token_learner_insertion_idx, self.num_layers):
            self.encoder.extend(
                [Block(dim, num_heads[idx]) for _ in range(num_blocks[idx])]
            )

        self.vq = VectorQuantize(
            dim=dim,
            codebook_dim=codebook_dim,
            codebook_size=num_codebook_embeddings,
            decay=decay, 
            commitment_weight=beta,
        )

        # Decoder Layers
        for idx in range(token_learner_insertion_idx, self.num_layers):
            self.decoder.extend(
                [Block(dim, num_heads[idx]) for _ in range(num_blocks[idx])]
            )

        self.decoder.append(TokenToSpatialTransformer(
            *self.init_patch_res, dim))

        for idx in range(token_learner_insertion_idx):
            self.decoder.extend(
                [Block(dim, num_heads[idx]) for _ in range(num_blocks[idx])]
            )

        self.patch_unembedding = PatchUnembedding(
            input_res, num_channels, dim, patch_size
        )
        self.apply(self.init_weights)

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
        return self.vq.forward(x_enc)  # Vector Quantizer

    def forward(self, img: torch.Tensor):
        x_enc = self.encode(img)  # Encoder
        z_q, indices, loss = self.quantize(x_enc)  # Vector Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
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
