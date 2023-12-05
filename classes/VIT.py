import math
import torch
from torch import nn
from torch.nn import functional
import os
import sys
sys.path.append(os.path.abspath('.'))
from classes.Transformers import Block, FeedFoward
from classes.SpectralNorm import SpectralNorm

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, image_size, patch_size, num_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(-2).transpose(-2, -1)
        return x

class Upscale(nn.Module):
    def __init__(self, num_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()

        self.upscale = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=num_channels,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor):
        B, C, D = x.shape
        x = x.transpose(1, 2) # B, D, C
        patch_dim = math.sqrt(C).__floor__()
        x = x.view(B, D, patch_dim, patch_dim)
        img = self.upscale(input = x)
        return img      
    
class VisionTransformerForClassification(nn.Module):

    
    def __init__(self, image_size: int, patch_size: int, num_channels: int, embed_dim: int, num_classes: int, NUM_BLOCKS = 3, dropout = 0):
        super().__init__()
    
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_patches = (self.image_size // self.patch_size) ** 2 # Num patches == Context
        self.mask = torch.zeros(size = (self.num_patches + 1, self.num_patches + 1)).bool().to(device=device)

        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, embed_dim).to(device=device)
        self.cls_token = nn.Parameter(data = torch.randn(size=(1,1, embed_dim), device=device))
        self.position_embedding_table = nn.Embedding(self.num_patches + 1, embed_dim, device=device)
        self.blocks = nn.ModuleList([Block(emb_dims=embed_dim, num_heads=4, dropout=dropout) for _ in range(NUM_BLOCKS)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim) 
        
        # Language model head used for output
        self.lm_head = nn.Linear(embed_dim, self.num_classes)


    def forward(self, x, targets=None):
        B, C, H, W = x.shape

        # x and targets are both (B,C) tensor of integers

        # Getting the Patch embeddings
        patch_emb: torch.Tensor = self.patch_embeddings(x) # (B,C,D)
        cls_token = self.cls_token.expand(B, -1, -1)

        # Added the Class token to the Patch embeddings
        x = torch.concat([cls_token, patch_emb], dim=1) # (B, C+1, D) Added Class token
        
        B, C, D = x.shape

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(torch.arange(C, device=device)) # (C,D)

        # Adding the position embedding to the patch embeddings 
        x = x + pos_emb

        for block in self.blocks:
            x = block(x, self.mask)
            
        x = self.ln_f(x) 
        logits = self.lm_head(x)
        cls_logits = logits[:, 0]

        if targets is None:
            loss = None
        else:
            loss = functional.cross_entropy(cls_logits, targets)

        return cls_logits, loss

    def predict(self, x):
        # Get the predictions
        cls_logits, loss = self.forward(x)
        probs = functional.softmax(cls_logits, dim=-1)
        predictions = probs.argmax(dim = -1)
        return predictions

class ViTEncoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_channels: int, embed_dim: int, num_heads: int, num_blocks: int, dropout: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.patch_embedding = PatchEmbeddings(image_size, patch_size, num_channels, embed_dim)
        
        scale = embed_dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(size = (1, num_patches, embed_dim)) * scale)
        self.pre_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(*[Block(embed_dim, num_heads, dropout) for _ in range(num_blocks)])
        
        self.initialize_weights()

    # Initializes all the layer weightd with a normalized value
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
        x = self.patch_embedding.forward(x)
        x = x + self.position_embedding
        x = self.pre_net_norm(x)
        x = self.transformer(x)
        
        return x

class ViTDecoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_channels: int, embed_dim: int, num_heads: int, num_blocks: int, dropout: int) -> None:
        super().__init__()
                
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        scale = embed_dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(size = (1, num_patches, embed_dim)) * scale)
        self.post_net_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.Sequential(*[Block(embed_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.proj = Upscale(num_channels, embed_dim, patch_size)    
    
        self.initialize_weights()


    def forward(self, x):
        x = x + self.position_embedding
        x = self.transformer(x)
        x = self.post_net_norm(x)
        x = self.proj(x)

        return x
    
    
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
    
class MultiHeadAttentionForViTDiscriminator(nn.Module):

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()

        # Key, Query and Value weights are (D, H)
        self.num_heads = num_heads
        self.head_size = head_size
        self.emb_dim = num_heads * head_size # Dimensionality 
        self.query = SpectralNorm(nn.Linear(self.emb_dim, self.emb_dim, bias=False))
        self.value = SpectralNorm(nn.Linear(self.emb_dim, self.emb_dim, bias=False))
        self.key =   SpectralNorm(nn.Linear(self.emb_dim, self.emb_dim, bias=False))
        self.value_proj = SpectralNorm(nn.Linear(self.emb_dim, self.emb_dim))# Additional layer for inter head communication

    def split(self, x:torch.Tensor):
        B, C, D = x.shape
        x = x.view(B, C, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3) # B, N, C, H

    def forward(self, query_input: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor, mask = None, output_attention = False):

        B, C, D = value_input.shape

        if mask is None:
            # Default mask is the encoder mask
            mask = torch.zeros(size = (C, C)).bool().to(device=value_input.device)

        # B, N, C, H in size
        query= self.split(self.query(query_input))
        key = self.split(self.key(key_input))
        value = self.split(self.value(value_input))

        wei = torch.cdist(query, key, p=2) * self.head_size ** -0.5 # (B, N, C, H) @ (B, N, H, C) => (B, N, C, C)

        wei = wei.masked_fill(mask, float('-inf')) # (B, N, C, C)
        wei = functional.softmax(wei, dim=-1) # (B, N, C, C)
        values = wei @ value # (B, N, C, C) @ (B, N, C, H) -> (B, N, C, H)
        values = values.permute(0, 2, 1, 3) # (B, C, N, H)
        values = values.reshape(B, C, self.emb_dim)
        values = self.value_proj(values)

        if output_attention:
            return values, wei

        return values

class ViTDiscriminatorBlock(nn.Module):

    def __init__(self, emb_dims, num_heads, dropout = 0):
        # emb_dims: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()

        # Divide the embedding dimensions by the number of heads to get the head size
        head_size = emb_dims // num_heads

        # Communication
        self.self_att = MultiHeadAttentionForViTDiscriminator(num_heads, head_size)

        # Computation
        self.feed_fwd = FeedFoward(emb_dims, dropout)

        # Adding Layer Normalization
        self.ln1 = nn.LayerNorm(emb_dims)
        self.ln2 = nn.LayerNorm(emb_dims)


    def forward(self, x, mask: torch.Tensor):
        # Residual connections allow the network to learn the simplest possible function. No matter how many complex layer we start by learning a linear function and the complex layers add in non linearity as needed to learn true function.
        x = x + self.self_att.forward(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.feed_fwd.forward(self.ln2(x))
        return x

class ViTDiscriminator(nn.Module):
       
    def __init__(self, image_size: int, patch_size: int, num_channels: int, embed_dim: int, num_classes: int, NUM_BLOCKS = 3, dropout = 0):
        super().__init__()
    
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_patches = (self.image_size // self.patch_size) ** 2 # Num patches == Context
        self.mask = torch.zeros(size = (self.num_patches + 1, self.num_patches + 1)).bool()

        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, embed_dim)
        self.cls_token = nn.Parameter(data = torch.randn(size=(1,1, embed_dim)))
        self.position_embedding_table = nn.Embedding(self.num_patches + 1, embed_dim)
        self.blocks = nn.ModuleList([ViTDiscriminatorBlock(emb_dims=embed_dim, num_heads=4, dropout=dropout) for _ in range(NUM_BLOCKS)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim) 
        
        # Projection head used for output
        self.proj = nn.Linear(embed_dim, self.num_classes)


    def forward(self, x, targets=None):
        B, C, H, W = x.shape

        # x and targets are both (B,C) tensor of integers

        # Getting the Patch embeddings
        patch_emb: torch.Tensor = self.patch_embeddings(x) # (B,C,D)
        cls_token = self.cls_token.expand(B, -1, -1)

        # Added the Class token to the Patch embeddings
        x = torch.concat([cls_token, patch_emb], dim=1) # (B, C+1, D) Added Class token
        
        B, C, D = x.shape

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(torch.arange(C)) # (C,D)

        # Adding the position embedding to the patch embeddings 
        x = x + pos_emb

        for block in self.blocks:
            x = block(x, self.mask)
            
        x = self.ln_f(x) 
        logits = self.proj(x)
        cls_logits = logits[:, 0]

        if targets is None:
            loss = None
        else:
            loss = functional.cross_entropy(cls_logits, targets)

        return cls_logits, loss

    def predict(self, x):
        # Get the predictions
        cls_logits, loss = self.forward(x)
        probs = functional.softmax(cls_logits, dim=-1)
        predictions = probs.argmax(dim = -1)
        return predictions
