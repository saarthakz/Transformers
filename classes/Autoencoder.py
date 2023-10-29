import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()    

        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=8,  kernel_size=(3,3), stride=2, padding=1), # -> N, 8, 16, 16
            nn.ReLU(),
            nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=(3,3), stride=2, padding=1), # -> N, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1), # -> N, 32, 4, 4
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1), # -> N, 64, 2, 2
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=2, padding=1), # -> N, 128, 1, 1
        )
        
        # N, 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=2, output_padding=1, padding=1), # -> N, 64, 2, 2
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=2, output_padding=1, padding=1), # -> N, 32, 4, 4
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=2, output_padding=1, padding=1), # -> N, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8,  kernel_size=(3,3), stride=2, output_padding=1, padding=1), # -> N, 8, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8,  out_channels=1,  kernel_size=(3,3), stride=2, output_padding=1, padding=1), # -> N, 1, 32, 32
            nn.Sigmoid(),
        )

        self.loss = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss = self.loss.forward(x, decoded)
        return decoded, loss
    