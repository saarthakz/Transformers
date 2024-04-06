import torch
import torch.nn as nn
from classes.VIT import (
    PatchEmbeddings,
    Block,
    Upscale,
    PoolDownsample,
    Upsample,
)
from classes.VectorQuantizer import VectorQuantizerEMA
from classes.Swin import res_scaler
import math


class ViT_PoolDownsample_BilinearUpsample(nn.Module):
    def __init__(
        self,
        dim=128,
        input_resolution: tuple[int] = (64, 64),
        patch_size=4,
        num_channels=3,
        num_codebook_embeddings=1024,
        codebook_dim=32,
        **kwargs
    ):
        super().__init__()

        encoder_layers = 2
        self.patch_embedding = PatchEmbeddings(num_channels, dim, patch_size)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        res = res_scaler(input_resolution, 1 / patch_size)
        self.init_patch_res = res

        for _ in range(encoder_layers):
            self.encoder.append(Block(dim, 4))
            self.encoder.append(Block(dim, 4))
            self.encoder.append(PoolDownsample(res))
            dim = dim * 2
            res = res_scaler(res, 0.5)

        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.vq = VectorQuantizerEMA(num_codebook_embeddings, codebook_dim, 0.5, 0.99)
        self.post_quant = nn.Linear(codebook_dim, dim)

        decoder_layers = 2

        for _ in range(decoder_layers):
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Block(dim, 4))
            self.decoder.append(Upsample(res, dim))

            dim = dim // 2
            res = res_scaler(res, 2)

        self.upscale = Upscale(num_channels, dim, patch_size)

    def forward(self, img: torch.Tensor):
        x = self.patch_embedding.forward(img)
        for idx, layer in enumerate(self.encoder):
            x = layer.forward(x)

        B, C, D = x.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x = x.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.vq.forward(x)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)

        for idx, layer in enumerate(self.decoder):
            z_q = layer.forward(z_q)

        z_q = self.upscale.forward(z_q, *self.init_patch_res)
        return z_q, indices, loss
