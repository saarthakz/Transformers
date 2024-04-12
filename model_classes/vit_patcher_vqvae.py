import torch
import torch.nn as nn
import os
import sys
from timm.models.layers import trunc_normal_

sys.path.append(os.path.abspath("."))
from classes.VIT import (
    PatchEmbedding,
    PatchEmbeddingOld,
    PatchUnembedding,
    Upscale,
    Block,
)
from classes.VectorQuantizer import VectorQuantizerEMA
from classes.Swin import res_scaler, MultiSwinBlock, PatchExpand, PatchMerge
from classes.SPT import ShiftedPatchEmbedding


# class ViT_PatchMergeExpand(nn.Module):
#     def __init__(
#         self,
#         dim=128,
#         input_res: list[int] = (64, 64),
#         patch_size=4,
#         num_channels=3,
#         num_codebook_embeddings=1024,
#         codebook_dim=32,
#         num_layers=2,
#         num_blocks=2,
#         with_swin=False,
#         num_heads=2,
#         swin_depth=2,
#         window_size=4,
#         with_shifted_patch_embeddings=False,
#         beta=0.25,
#         decay=0.01,
#         **kwargs,
#     ):
#         super().__init__()

#         self.with_swin = with_swin
#         self.with_shifted_patch_embeddings = with_shifted_patch_embeddings
#         self.num_layers = num_layers
#         self.num_blocks = num_blocks
#         self.patch_embedding = (
#             ShiftedPatchEmbedding(num_channels, dim, patch_size)
#             if with_shifted_patch_embeddings
#             else PatchEmbedding(num_channels, dim, patch_size)
#         )
#         self.encoder = nn.ModuleList()
#         self.decoder = nn.ModuleList()

#         res = res_scaler(input_res, 1 / patch_size)
#         self.init_patch_res = res

#         # Encoder Layers
#         for _ in range(self.num_layers):
#             if with_swin:
#                 self.encoder.append(
#                     MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
#                 )
#             for _ in range(self.num_blocks):
#                 self.encoder.append(Block(dim, num_heads))

#             self.encoder.append(PatchMerge(res, dim, dim * 2, 2))
#             dim = dim * 2
#             res = res_scaler(res, 0.5)

#         self.pre_quant = nn.Linear(dim, codebook_dim)
#         self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, beta, decay)
#         self.post_quant = nn.Linear(codebook_dim, dim)

#         # Decoder Layers
#         for _ in range(self.num_layers):
#             if with_swin:
#                 self.decoder.append(
#                     MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
#                 )
#             for _ in range(self.num_blocks):
#                 self.decoder.append(Block(dim, num_heads))

#             self.decoder.append(PatchExpand(res, dim, dim // 2, 2))
#             dim = dim // 2
#             res = res_scaler(res, 2)

#         self.patch_unembedding = PatchUnembedding(
#             input_res,
#             num_channels,
#             dim,
#             patch_size,
#         )
#         self.apply(self._init_weights)

#     def encode(self, x: torch.Tensor):
#         x = self.patch_embedding.forward(x)
#         for layer in self.encoder:
#             x = layer.forward(x)

#         x = self.pre_quant.forward(x)
#         return x

#     def decode(self, z_q: torch.Tensor):
#         z_q = self.post_quant.forward(z_q)
#         for layer in self.decoder:
#             z_q = layer.forward(z_q)

#         z_q = self.patch_unembedding.forward(z_q)
#         return z_q

#     def quantize(self, x_enc: torch.Tensor):
#         B, C, D = x_enc.shape
#         patch_H, patch_W = res_scaler(self.init_patch_res, 1 / (2**self.num_layers))

#         assert (
#             C == patch_H * patch_W
#         ), f"Input patch length {C} does not match the patch resolution {patch_H} {patch_W}"
#         x_enc = x_enc.transpose(-2, -1).view(B, D, patch_H, patch_W)
#         z_q, indices, loss = self.vq.forward(x_enc)  # Vector Quantizer
#         z_q = z_q.view(B, D, C).transpose(-2, -1)
#         return z_q, indices, loss

#     def forward(self, img: torch.Tensor):
#         x_enc = self.encode(img)  # Encoder
#         z_q, indices, loss = self.quantize(x_enc)  # Vector Quantizer
#         recon_imgs = self.decode(z_q)  # Decoder
#         return recon_imgs, indices, loss

#     def get_recons(self, x: torch.Tensor):
#         recon_imgs, _, _ = self.forward(x)
#         return recon_imgs


#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#             if m.weight is not None:
#                 nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.xavier_normal_(m.weight, gain=0.02)
#             if hasattr(m, "bias") and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
class ViT_PatchMergeExpand(nn.Module):
    def __init__(
        self,
        dim=128,
        input_res: list[int] = [64, 64],
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        num_layers=2,
        with_swin=False,
        num_heads=2,
        swin_depth=2,
        window_size=4,
        with_shifted_patch_embeddings=False,
        **kwargs,
    ):
        super().__init__()

        self.with_swin = with_swin
        self.with_shifted_patch_embeddings = with_shifted_patch_embeddings
        self.num_layers = num_layers
        self.patch_embedding = (
            ShiftedPatchEmbedding(num_channels, dim, patch_size)
            if with_shifted_patch_embeddings
            else PatchEmbeddingOld(num_channels, dim, patch_size)
        )
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for _ in range(self.num_layers):
            self.encoder.append(Block(dim, 4))
            if with_swin:
                self.encoder.append(
                    MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
                )
            self.encoder.append(PatchMerge(res, dim, dim * 2, 2))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        # Decoder Layers
        for _ in range(self.num_layers):
            self.decoder.append(Block(dim, 4))
            if with_swin:
                self.decoder.append(
                    MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
                )
            self.decoder.append(PatchExpand(res, dim, dim // 2, 2))
            dim = dim // 2
            res = res_scaler(res, 2)

        self.upscale = Upscale(self.init_patch_res, num_channels, dim, patch_size)
        self.apply(self._init_weights)

    def encode(self, x: torch.Tensor):
        x = self.patch_embedding.forward(x)
        for layer in self.encoder:
            x = layer.forward(x)
        return x

    def decode(self, z_q: torch.Tensor):
        for layer in self.decoder:
            z_q = layer.forward(z_q)

        z_q = self.upscale.forward(z_q)
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


class ViT_PatchMergeExpand_Old(nn.Module):
    def __init__(
        self,
        dim=128,
        input_res: list[int] = (64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        num_layers=2,
        num_blocks=2,
        with_swin=False,
        num_heads=2,
        swin_depth=2,
        window_size=4,
        with_shifted_patch_embeddings=False,
        beta=0.25,
        decay=0.01,
        **kwargs,
    ):
        super().__init__()

        self.with_swin = with_swin
        self.with_shifted_patch_embeddings = with_shifted_patch_embeddings
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.patch_embedding = (
            ShiftedPatchEmbedding(num_channels, dim, patch_size)
            if with_shifted_patch_embeddings
            else PatchEmbeddingOld(num_channels, dim, patch_size)
        )
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_res, 1 / patch_size)
        self.init_patch_res = res

        # Encoder Layers
        for _ in range(self.num_layers):
            if with_swin:
                self.encoder.append(
                    MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
                )
            for _ in range(self.num_blocks):
                self.encoder.append(Block(dim, num_heads))

            self.encoder.append(PatchMerge(res, dim, dim * 2, 2))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, beta, decay)
        self.post_quant = nn.Linear(codebook_dim, dim)

        # Decoder Layers
        for _ in range(self.num_layers):
            if with_swin:
                self.decoder.append(
                    MultiSwinBlock(dim, res, swin_depth, num_heads, window_size)
                )
            for _ in range(self.num_blocks):
                self.decoder.append(Block(dim, num_heads))

            self.decoder.append(PatchExpand(res, dim, dim // 2, 2))
            dim = dim // 2
            res = res_scaler(res, 2)

        self.patch_unembedding = Upscale(
            self.init_patch_res,
            num_channels,
            dim,
            patch_size,
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
