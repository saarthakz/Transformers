import math
import torch
from torch import nn
from torch.nn import functional
from Transformers import Block

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
        x = x.flatten(2).transpose(1, 2)
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

# %%
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

# %%
class VisionTransformerAutoencoder(nn.Module):

    def __init__(self, image_size: int, patch_size: int, num_channels: int, embed_dim: int, num_classes: int, NUM_BLOCKS = 3, dropout = 0):
        super().__init__()
    
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_patches = (self.image_size // self.patch_size) ** 2 # Num patches == Context
        self.mask = torch.zeros(size = (self.num_patches, self.num_patches)).bool().to(device=device)

        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, embed_dim).to(device=device)
        self.position_embedding_table = nn.Embedding(self.num_patches, embed_dim, device=device)
        self.blocks = nn.ModuleList([Block(emb_dims=embed_dim, num_heads=4, dropout=dropout) for _ in range(NUM_BLOCKS)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim) 
        self.upscale = Upscale(num_channels=num_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.mse = nn.MSELoss()
        
    def forward(self, x, targets=None):
        B, C, H, W = x.shape

        # x and targets are both (B,C) tensor of integers

        self.img = x # Saving the image for calculating reconstruction loss later

        # Getting the Patch embeddings
        x : torch.Tensor = self.patch_embeddings(x) # (B,C,D)

        B, C, D = x.shape

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(torch.arange(C, device=device)) # (C,D)

        # Adding the position embedding to the patch embeddings 
        x = x + pos_emb

        for block in self.blocks:
            x = block(x, self.mask)
            
        x = self.ln_f(x)
        img = self.upscale(x)
        loss = self.mse(img, self.img) # Inputs, targets
        return img, loss
    