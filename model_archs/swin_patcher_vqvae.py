import torch
import torch.nn as nn
import os
import sys
from timm.models.layers import trunc_normal_

sys.path.append(os.path.abspath("."))
from classes.VIT import (
    PatchEmbedding,
    PatchUnembedding,
)
from vector_quantize_pytorch import VectorQuantize

from classes.Swin import res_scaler, MultiSwinBlock, PatchExpand, PatchMerge
from classes.SPT import ShiftedPatchEmbedding


class Model(nn.Module):
    def __init__(
        self,
        input_res: list[int] = [64, 64],
        patch_size=4,
        num_channels=3,
        dim=128,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        beta=0.5,
        decay=0.99,
        swin_depths: list[int] = [2, 2, 2, 2],
        num_heads: list[int] = [3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        with_shifted_patch_embeddings=False,
        **kwargs,
    ):
        super().__init__()

        self.with_shifted_patch_embeddings = with_shifted_patch_embeddings
        self.patch_embedding = (
            ShiftedPatchEmbedding(num_channels, dim, patch_size)
            if with_shifted_patch_embeddings
            else PatchEmbedding(num_channels, dim, patch_size)
        )
        self.num_layers = len(num_heads)
        self.swin_depths = swin_depths
        self.num_heads = num_heads
        self.ape = ape
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # dpr = [
        #     x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_depths))
        # ]  # stochastic depth decay rule

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.init_patch_res[0] * self.init_patch_res[1], dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Encoder Layers
        for idx, depth in enumerate(self.swin_depths):
            self.encoder.append(
                MultiSwinBlock(
                    dim,
                    res,
                    depth,
                    num_heads[idx],
                    window_size,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    # dpr[sum(swin_depths[:idx]) : sum(swin_depths[: idx + 1])],
                    norm_layer,
                )
            )

            self.encoder.append(PatchMerge(res, dim))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        # Vector Quantizer
        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantize(
            codebook_dim, num_codebook_embeddings, decay=decay, commitment_weight=beta
        )
        self.post_quant = nn.Linear(codebook_dim, dim)

        self.num_heads.reverse()
        self.swin_depths.reverse()

        # Decoder Layers
        for idx, depth in enumerate(self.swin_depths):
            self.decoder.append(
                MultiSwinBlock(
                    dim,
                    res,
                    depth,
                    num_heads[idx],
                    window_size,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    # dpr[sum(swin_depths[:idx]) : sum(swin_depths[: idx + 1])],
                    norm_layer,
                )
            )
            self.decoder.append(PatchExpand(res, dim))
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

        x = self.pre_quant.forward(x)
        return x

    def decode(self, z_q: torch.Tensor):
        z_q = self.post_quant.forward(z_q)
        for layer in self.decoder:
            z_q = layer.forward(z_q)

        z_q = self.patch_unembedding.forward(z_q)
        return z_q

    def quantize(self, x_enc: torch.Tensor):
        B, C, D = x_enc.shape
        patch_H, patch_W = res_scaler(self.init_patch_res, 1 / (2**self.num_layers))

        assert (
            C == patch_H * patch_W
        ), f"Input patch length {C} does not match the patch resolution {patch_H} {patch_W}"
        x_enc = x_enc.transpose(-2, -1).view(B, D, patch_H, patch_W)
        z_q, indices, loss = self.vq.forward(x_enc)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
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
