import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from classes.Transformers import GPT
from classes.Swin import res_scaler


class Model(nn.Module):
    def __init__(
        self,
        vq_model: nn.Module,
        dim: int,
        num_heads: int,
        num_codebook_embeddings: int,
        keep_prob=0.8,
        **kwargs,
    ):
        super().__init__()
        self.keep_prob = keep_prob
        self.embed_dim = dim
        self.num_codebook_embeddings = num_codebook_embeddings
        self.num_heads = num_heads
        self.vq_model = vq_model
        self.patch_res = res_scaler(
            self.vq_model.init_patch_res, 1 / (2**self.vq_model.num_layers)
        )
        H, W = self.patch_res
        self.num_patches = H * W

        self.transformer = GPT(
            context=512,
            emb_dims=dim,
            vocab_size=num_codebook_embeddings,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor):
        x_enc = self.vq_model.encode(x)
        B, C, D = x_enc.shape
        z_q, indices, loss = self.vq_model.quantize(x_enc)

        # print(z_q.shape, indices.shape)

        # Indices will be fed to the transformer for prediction
        indices = indices.view(B, -1)
        # print(indices.shape)

        # Base indices are also the target for when predicting from noisy indices
        target = indices

        # Start token so that the Transformer always has a token when generating
        start_tokens = (
            torch.ones(size=[B, 1], dtype=torch.long, device=indices.device) * 0
        )

        mask = torch.bernoulli(self.keep_prob * torch.ones(indices.shape)).to(
            indices.device
        )
        mask = mask.round().to(dtype=torch.int64)

        random_indices = torch.randint_like(
            indices, low=0, high=self.num_codebook_embeddings
        )

        noised_indices = mask * indices + (1 - mask) * random_indices
        noised_indices = torch.cat((start_tokens, noised_indices), dim=1)[:, :-1]

        logits, loss = self.transformer.forward(noised_indices, target)

        return logits, loss

    @torch.no_grad()
    def sample(self, num_samples=16):

        device = f"cuda:{torch.cuda.current_device()}"

        start_tokens = (
            torch.ones(size=[num_samples, 1], dtype=torch.long, device=device) * 0
        )

        indices = self.transformer.generate(start_tokens, 512)[:, 1:]
        indices = indices.reshape(-1, 1)

        encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=device)
        encodings.scatter_(1, indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(
            size=(num_samples, patch_H * patch_W, 256)
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        patch_H, patch_W = res_scaler(
            self.vq_model.init_patch_res, 1 / (2**self.vq_model.num_layers)
        )

        # z_q = self.vq_model.vq.embedding.forward(indices).view(
        #     size=(num_samples, patch_H * patch_W, 256)
        # )

        # z_q = z_q.permute(0, 3, 1, 2)  # (B, D, H, W)

        # z_q = z_q.view(
        #     num_samples,
        #     self.embed_dim,
        # ).transpose(
        #     -2, -1
        # )  # (B, D, C) -> [after transpose] -> (B, C, D)
        recon_imgs = self.vq_model.decode(z_q)
        return recon_imgs
