import torch
from torch import nn
import torch.nn.functional as func
import os
import sys

sys.path.append(os.path.abspath("."))
from classes.VectorQuantizer import VectorQuantizer, VectorQuantizerEMA


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=res_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # This Conv2D does not change the shape of the input, only the channels
            nn.ReLU(),
            nn.Conv2d(
                in_channels=res_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # This Conv2D does not change the shape of the input, including the channels. Try using the Conv Attention Block here instead
        )

        if in_channels != out_channels:
            self.channel_match = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
            )  # Used for channel matching the 'residual' input

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_match(x) + self.net(x)
        else:
            return x + self.net(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        num_residual_layers: int,
    ):
        super(ResidualStack, self).__init__()
        layers = [
            ResBlock(in_channels, res_channels, out_channels)
            for _ in range(num_residual_layers)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return func.relu(self.net.forward(x))


class UpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(4, 4),
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.upsample(x)


# Only changes the number of channels
class Channelizer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1
        )

    def forward(self, x):
        return self.net.forward(x)


class DownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down_sample = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.down_sample(x)


# Non Local Block
class ConvAttention(nn.Module):
    def __init__(self, channels: int, input_res: tuple[int]) -> None:
        super().__init__()

        self.in_channels = channels
        self.input_res = input_res

        self.layer_norm = nn.LayerNorm(channels)

        self.query = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )
        self.key = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )
        self.value = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )

        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        gn_x = self.layer_norm(x)

        H, W = self.input_res
        B, T, C = gn_x.shape
        assert (
            T == H * W
        ), "Input resolution doesn't match the context of the input resolution"
        gn_x = gn_x.view(B, H, W, C)
        gn_x = gn_x.permute(0, 3, 1, 2)

        query = self.query(gn_x).reshape(B, C, H * W)
        query = query.permute(0, 2, 1)
        key = self.key(gn_x).reshape(B, C, H * W)
        value = self.value(gn_x).reshape(B, C, H * W)

        attn_score = query @ key * C**-0.5
        # compute attention scores ("affinities")

        attn_score = func.softmax(attn_score, dim=-1)  # (B, T, T)

        attn_score = (value @ attn_score).transpose(-2, -1)

        return x + attn_score


# Encoder as implemented in the original VQGAN Paper
class Encoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_dim: int,
        res_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(Channelizer(image_channels, latent_dim // 2))
        layers.append(DownSample(latent_dim // 2))
        layers.append(nn.ReLU())

        layers.append(Channelizer(latent_dim // 2, latent_dim))
        layers.append(DownSample(latent_dim))
        layers.append(nn.ReLU())

        layers.append(Channelizer(latent_dim, latent_dim))

        layers.append(
            ResidualStack(latent_dim, latent_dim, res_channels, num_residual_layers)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Decoder as implemented in the original VQGAN Paper
class Decoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_dim: int,
        res_channels: int,
        num_residual_layers: int,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(Channelizer(latent_dim, latent_dim))
        layers.append(
            ResidualStack(latent_dim, latent_dim, res_channels, num_residual_layers)
        )
        layers.append(UpSample(latent_dim))
        layers.append(nn.ReLU())

        layers.append(Channelizer(latent_dim, latent_dim // 2))
        layers.append(UpSample(latent_dim // 2))
        layers.append(nn.ReLU())

        layers.append(Channelizer(latent_dim // 2, image_channels))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        image_channels: int,
        latent_dim: int,
        res_channels: int,
        num_residual_layers: int,
        vq_num_embeddings: int,
        beta: int,
        decay=0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            image_channels, latent_dim, res_channels, num_residual_layers
        )
        self.decoder = Decoder(
            image_channels, latent_dim, res_channels, num_residual_layers
        )
        self.codebook = (
            VectorQuantizer(vq_num_embeddings, latent_dim, beta)
            if decay == 0
            else VectorQuantizerEMA(vq_num_embeddings, latent_dim, beta, decay)
        )

        self.pre_quant_conv = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )

        self.post_quant_conv = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
        )

    def encode(self, x):
        encoded = self.encoder(x)
        pre_quant_encoded = self.pre_quant_conv(encoded)
        return pre_quant_encoded

    def through_codebook(self, pre_quant_encoded):
        codebook_mapping, codebook_indices, q_loss = self.codebook(pre_quant_encoded)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_encoded = self.post_quant_conv(z)
        decoded = self.decoder(post_quant_encoded)
        return decoded

    # Combining the Encoder, Codebook and Decoder methods
    def forward(self, x):
        pre_quant_encoded = self.encode(x)
        codebook_mapping, codebook_indices, q_loss = self.through_codebook(
            pre_quant_encoded
        )
        decoded = self.decode(codebook_mapping)

        return decoded, codebook_indices, q_loss

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
