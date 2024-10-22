import math
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from vector_quantize_pytorch import VectorQuantize as VQ
from classes.VIT import Block, PatchEmbedding, PatchUnembedding
from classes.Swin import res_scaler
from einops import repeat, pack, unpack


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: list[int],
        dropout: int,
    ) -> None:
        super().__init__()

        self.pre_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(
            *[Block(embed_dim, heads, dropout) for heads in num_heads]
        )

        self.initialize_weights()

    # Initializes all the layer weights with a normalized value
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pre_net_norm(x)
        x = self.transformer(x)

        return x


class Model(nn.Module):
    """
    Args:
        input_res (list[int]): Image size as (H, W),
        dim (int): Initial Patch embedding dimension,
        patch_size (int): Patch size,
        num_channels (int): Number of input image channels,
        num_latent_tokens (int): Number of latent tokens to be used in TiTok
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
        input_res: list[int],
        dim=128,
        patch_size=4,
        num_channels=3,
        num_latent_tokens=32,
        num_codebook_embeddings=1024,
        codebook_dim=128,
        num_heads: list[int] = [4, 4, 4],
        dropout=0,
        beta=0.25,
        decay=0.99,
        **kwargs,
    ):
        super().__init__()

        self.input_res = input_res
        self.patch_size = patch_size
        self.patch_res = res_scaler(self.input_res, 1 / self.patch_size)
        patch_H, patch_W = self.patch_res
        num_patches = patch_H * patch_W

        self.encoder = Encoder(
            dim,
            num_heads,
            dropout,
        )
        self.decoder = Encoder(
            dim,
            num_heads,
            dropout,
        )
        self.image_to_embedding = PatchEmbedding(num_channels, dim, patch_size)
        self.embedding_to_image = PatchUnembedding(
            self.input_res, num_channels, dim, patch_size
        )
        self.num_latent_tokens = num_latent_tokens

        self.position_embedding = nn.Parameter(torch.zeros(size=[1, num_patches, dim]))
        self.latent_tokens = nn.Parameter(torch.zeros(size=[num_latent_tokens, dim]))
        self.mask_tokens = nn.Parameter(torch.zeros(size=[num_patches, dim]))

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.latent_tokens, std=0.02)
        nn.init.normal_(self.mask_tokens, std=0.02)

        self.quantizer = VQ(
            dim=dim,
            codebook_size=num_codebook_embeddings,
            codebook_dim=codebook_dim,
            decay=decay,
            commitment_weight=beta,
        )
        self.pre_quant = nn.Linear(dim, codebook_dim)
        self.post_quant = nn.Linear(codebook_dim, dim)

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x: torch.Tensor):
        batch = x.shape[0]
        tokens = self.image_to_embedding(x)
        tokens = self.position_embedding + tokens

        # Appending the latents
        latents = repeat(self.latent_tokens, "l d -> b l d", b=batch)
        packed, packed_shape = pack([tokens, latents], "b * d")
        packed = self.encoder(packed)

        # Extract latents
        _, latents = unpack(packed, packed_shape, "b * d")
        latents = self.pre_quant(latents)
        return latents

    def decode(self, latents: torch.Tensor):
        latents = self.post_quant(latents)

        # Append the masking tokens here
        batch = latents.shape[0]
        mask_tokens = repeat(self.mask_tokens, "n d -> b n d", b=batch)
        packed, packed_shape = pack([mask_tokens, latents], "b * d")
        packed = self.decoder(packed)

        # Extract the tokens
        tokens, _ = unpack(packed, packed_shape, "b * d")

        x = self.embedding_to_image(tokens)
        # x = x.clamp(-1.0, 1.0)
        return x

    """
    def quantize(self, x_enc: torch.Tensor):
        B, C, D = x_enc.shape
        patch_H, patch_W = self.patch_res

        assert (
            C == patch_H * patch_W
        ), "Input patch length does not match the patch resolution"

        x_enc = x_enc.transpose(-2, -1).view(B, D, patch_H, patch_W)
        z_q, indices, loss = self.quantizer.forward(x_enc)  # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
        return z_q, indices, loss
    """

    def forward(self, x: torch.Tensor):
        x_enc = self.encode(x)  # Encoder
        z_q, indices, loss = self.quantizer.forward(x_enc)  # Vector Quantizer
        recon_imgs = self.decode(z_q)  # Decoder
        return recon_imgs, indices, loss

    def get_recons(self, x: torch.Tensor):
        recon_imgs, _, _ = self.forward(x)
        return recon_imgs
