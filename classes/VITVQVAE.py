import math
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath("."))
from classes.VectorQuantizer import VectorQuantizer
from classes.VIT import ViTEncoder, ViTDecoder

class ViTVQVAE(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_channels: int, num_embeddings: int, embed_dim: int, num_heads: int, num_blocks: int, dropout: int, beta: int):
        super().__init__()

        self.encoder = ViTEncoder(image_size, patch_size, num_channels, embed_dim, num_heads, num_blocks, dropout)
        self.decoder = ViTDecoder(image_size, patch_size, num_channels, embed_dim, num_heads, num_blocks, dropout)
        self.quantize = VectorQuantizer(num_embeddings, embed_dim, beta)
        self.pre_quant = nn.Linear(embed_dim, embed_dim)
        self.post_quant = nn.Linear(embed_dim, embed_dim)  
            
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.pre_quant(x)
        return x
    
    def decode(self, x):
        x = self.post_quant(x)
        x = self.decoder(x)
        return x.clamp(-1.0, 1.0)
    
    def forward(self, x):
        x_enc = self.encode(x) # Encoder
        B, C, D  = x_enc.shape
        num_patch_sqrt = math.sqrt(C).__int__()
        x_enc = x_enc.transpose(-2, -1).view(B, D, num_patch_sqrt, num_patch_sqrt)
        z_q, indices, loss = self.quantize.forward(x_enc) # Vector Quantizer
        z_q = z_q.view(B, D, C).transpose(-2, -1)
        recon_img: torch.Tensor = self.decode(z_q) # Decoder
        return recon_img, indices, loss
    
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path))
    
