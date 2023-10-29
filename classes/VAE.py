import torch
import torch.nn as nn
from torch.nn import functional
from typing import Union

class VAE(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: Union[list[int], None] = None, latent_dim = 4):
        super().__init__()    

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims if hidden_dims is not None else [64, 128]
        modules = []

        # Encoder modules
        for hidden_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=(3,3), stride=2, padding=1),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))    
            in_channels = hidden_dim

        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        ###

        # Latent representatiom layers
        self.mean = nn.Linear(in_features=self.hidden_dims[-1] * 8 * 8, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=self.hidden_dims[-1] * 8 * 8, out_features=latent_dim)
        self.latent_to_decoder = nn.Linear(in_features=latent_dim, out_features=self.hidden_dims[-1] * 8 * 8)
        ###

        # Decoder modules
        modules = [nn.Unflatten(1, (self.hidden_dims[-1], 8, 8))]
        self.hidden_dims.reverse()
        self.hidden_dims.append(self.in_channels)
        self.hidden_dims.pop(0)

        for hidden_dim in self.hidden_dims:
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=(4,4), stride=2, padding=1),
                # nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))    
            in_channels = hidden_dim

        # Output layer
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, kernel_size=(3,3), stride=1, padding=1),
            nn.Sigmoid()
        ))
        self.decoder = nn.Sequential(*modules)
        ###

    def forward(self, x, kl_weight = 2.5e-4):
        encoded = self.encoder(x) # Encoded input
        mean = self.mean(encoded) # Mean 
        log_var = self.log_var(encoded) # Log of variance
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)

        z = mean + std * epsilon # Latent representation as a sample from a distribution. Shape of z is [B, latent_dim]

        decoder_inp = self.latent_to_decoder(z) # Decoder input
        decoded = self.decoder(decoder_inp) # Decoder output

        recon_loss = functional.binary_cross_entropy(decoded, x, reduction='sum') # Reconstruction loss

        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # Distribution loss to push towards the Standard Normal Distribution

        loss = recon_loss + kl_weight * kl_div

        return decoded, loss
