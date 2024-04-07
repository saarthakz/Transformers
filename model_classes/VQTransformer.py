import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from classes.Transformers import Transformer


class VQTransformer(nn.Module):
    def __init__(
        self,
        vq_model: nn.Module,
        num_patches: int,
        embed_dim: int,
        num_embeddings: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__()
        self.transformer = Transformer(
            context=num_patches,
            emb_dims=embed_dim,
            vocab_size=num_embeddings,
            num_heads=num_heads,
        )

    @torch.no_grad()
    def vq_encode(self, x: torch.Tensor):
        self.vq_model.encode
