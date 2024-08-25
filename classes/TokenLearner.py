import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses an MLP with GELU in between. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    Attributes:
        num_tokens: Number of tokens.
        embed_dim: The size of hidden units in the MLP for spatial attention.
        dropout_rate: Dropout rate.
    """

    def __init__(self, num_tokens, embed_dim=64, dropout_rate=0.0):
        super(TokenLearner, self).__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # Define the layers
        self.layer_norm = nn.LayerNorm(
            embed_dim
        )  # We will calculate the normalized shape dynamically
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, num_tokens),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_tokens, num_tokens),
        )

    def forward(self, inputs):
        """Applies learnable tokenization to the 2D inputs.

        Args:
            inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
            Output of shape `[bs, n_token, c]`.
        """
        if inputs.dim() == 4:
            n, h, w, c = inputs.size()
            inputs = inputs.view(n, h * w, c)

        feature_shape = inputs.size()

        # Apply LayerNorm
        selected = self.layer_norm(inputs)

        # Apply the MLP block
        selected = self.mlp(selected)

        # Reshape and apply softmax
        selected = selected.view(
            feature_shape[0], -1, self.num_tokens
        )  # Shape: [bs, h*w, n_token]
        selected = selected.permute(0, 2, 1)  # Shape: [bs, n_token, h*w]
        selected = F.softmax(selected, dim=-1)

        # Apply einsum operation to get the final output
        feat = inputs.view(
            feature_shape[0], -1, feature_shape[-1]
        )  # Shape: [bs, h*w, c]
        feat = torch.einsum("bsi,bic->bsc", selected, feat)

        return feat


class TokenToSpatialTransformer(nn.Module):
    def __init__(self, height, width, dim=64, dropout_rate=0.0):
        super(TokenToSpatialTransformer, self).__init__()
        self.height = height
        self.width = width
        self.dim = dim

        # Initialize learnable spatial grid embeddings
        self.spatial_embeddings = nn.Parameter(torch.randn(1, height * width, dim))

        # Self-attention mechanism with batch_first=True
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=8, batch_first=True
        )

        # Linear projection to map tokens to spatial features

        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens):
        batch_size = tokens.size(0)

        # Repeat spatial embeddings for batch
        spatial_embeds = self.spatial_embeddings.repeat(
            batch_size, 1, 1
        )  # Shape: (batch, height * width, dim)

        # Apply attention: tokens attending to spatial grid
        attended_tokens, _ = self.attention(
            query=spatial_embeds, key=tokens, value=tokens
        )

        # Linear projection
        return self.proj(attended_tokens)  # Shape: (batch, height * width, dim)
