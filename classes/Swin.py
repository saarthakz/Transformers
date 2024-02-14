# %% [markdown]
# > # References
# 
# This implementation is based on the official [Microsoft implementation](https://github.com/microsoft/Swin-Transformer) with some modifications and annotations.

# %%
import torch
import torch.nn as nn
from classes.VIT import PatchEmbeddings
from typing import Union

# %% [markdown]
# ## Functions

# %%
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
        
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
        
    """
    
    B, H, W, C = x.shape
    
    # Convert to (B, window_height_count, window_size, window_width_count, window_size, C) 
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    
    # Convert to (B, window_height_count, window_width_count, window_size,  window_size, C) 
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() 
    
    # Efficient Batch Computation - Convert to (B * window_height_count * window_width_count, window_size,  window_size, C)
    # num_windows = window_height_count * window_width_count
    # Combining all the windows together with the batch dimension
    windows = windows.view(-1, window_size, window_size, C)
    
    return windows

def window_combination(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size (int): 
        H (int): Height of image (patch-wise)
        W (int): Width of image (patch-wise)
        
    Returns:
        x: (B, H, W, C)
    """
    
    # Get B from num_windows * B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    
    # Convert to (B * window_height_count * window_width_count, window_size,  window_size, C)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    
    # Convert to (B, window_height_count, window_size, window_width_count, window_size, C) Convert to (B, 8, 7, 8, 7, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Convert to (B, H, W, C)
    x = x.view(B, H, W, -1)
    
    return x

# %% [markdown]
# ## Modules

# %%
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features=Union[None|int], out_features=Union[None|int], act_layer=nn.GELU, drop=0.):
        super().__init__()
  
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act_layer.forward(x)
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)
        x = self.dropout.forward(x)
        return x

# %%
class PatchMerging(nn.Module):
    """ 
    Patch Merging Layer from the paper (Down Sampling)
    
    This merges the patch through a positional selection process

    Args:
        input_resolution (tuple[int]): Resolution of input feature
        dim (int): Number of input channels. (C)
        norm_layer (nn.Module, optional): Normalization layer. (Default: nn.LayerNorm)
    """
    
    class PatchMerging(nn.Module):
        def __init__(self, in_channels, out_channels, downscaling_factor):
            super().__init__()
            self.downscaling_factor = downscaling_factor
            self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor)
            self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

        def forward(self, x):
            b, c, h, w = x.shape
            new_h = h // self.downscaling_factor
            new_w = w // self.downscaling_factor
            x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
            return x # Shape of X: B, H/2, W/2, 2C

# %%
class WindowAttention(nn.Module):
    """ 
    Window based multi-head self attention(W-MSA) module with relative position bias.\n
    Used as Shifted-Window Multi-head self-attention(SW-MSA) by providing shift_size parameter in SwinTransformerBlock module
        
    Args:
        dim (int): Number of input channels (C)
        window_size int: The height and width of the window (M)
        num_heads (int): Number of attention heads for multi-head attention
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight (Default: 0.0)
        proj_drop (float, optional): Dropout ratio of output (Default: 0.0)
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size # window_height = window_width = M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
    
        # Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # W_Q, W_K, W_V
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, x: torch.Tensor, mask=Union[None|torch.Tensor]):
        """
        Args:
            x: input features with shape of (num_windows * B, N, C), N refers to number of patches in a window (M^2)
            mask: (0/-inf) mask with shape of (num_windows, M^2, M^2) or None -> 0 means applying attention, -inf means removing attention
        """
        # (batch, M^2, C)
        B_W, N, C = x.shape
        
        # (num_windows * B, N, 3C)
        qkv = self.qkv.forward(x)
        
        # (B_W, N, 3, num_heads, C // num_heads)
        qkv = qkv.reshape(B_W, N, 3, self.num_heads, C // self.num_heads)
        
        # Permute to (3, B, num_heads, N, C // num_heads)
        '''
        3: referring to q, k, v (total 3)
        B: batch size
        num_heads: multi-headed attention
        N:  M^2, referring to each token(patch)
        C // num_heads: Each head of each of (q,k,v) handles C // num_heads -> match exact dimension for multi-headed attention
        '''
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Decompose to query/key/vector for attention
        # each of q, k, v has dimension of (B, num_heads, N, C // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # Why not tuple-unpacking?
        
        q = q * self.scale
        
        # attn becomes (B_W, num_heads, N, N) shape
        # N = M^2
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_windows = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0) # (1, num_windows, 1, M^2, M^2), Unsqueeze 1 to broadcast along all heads and unsqueeze(0) to broadcast along all batches
            attn = attn.view(B_W // num_windows, num_windows, self.num_heads, N, N) # (B, num_windows, num_heads, N, N)
            attn = attn + mask
            
            
            attn = attn.view(-1, self.num_heads, N, N) # attn = (num_windows * B, num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        attn = self.attn_drop(attn)
        
        # attn = (num_windows * B, num_heads, N, N)
        # v = (B_W, num_heads, N, C // num_heads) B_W = num_windows * B
        # attn @ v = (num_windows * B, num_heads, N, C // num_heads)
        # (attn @ v).transpose(1, 2) = (num_windows * B, N, num_heads, C // num_heads)
        # Finally, x = (num_windows*B, N, C), reshape(B_, N, C) performs concatenation of multi-headed attentions
        x = (attn @ v).transpose(1, 2).reshape(B_W, N, C)
        
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x        

# %%
class SingleSwinBlock(nn.Module):
    """ Swin Transformer Block. It's used as either W-MSA or SW-MSA depending on shift_size
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution, after patchifying 
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float):Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer(nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): NOrmalization layer. Default: nn.LayerNorm
    """
    
    def __init__(
            self, 
            dim: int, 
            input_resolution: tuple[int], 
            num_heads: int, 
            window_size=4, 
            shift_size=0,
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None, 
            drop=0., 
            attn_drop=0., 
            act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm
        ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # If window_size > input_resolution, no partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        
        # Attention
        self.attn = WindowAttention(
            dim, 
            window_size=self.window_size, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        self.norm2 = norm_layer(dim)
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer, 
            drop=drop
        )
        
        # Attention Mask for SW-MSA
        # This handling of attention-mask is my favorite part. What a beautiful implementation.
        if self.shift_size > 0:
            H, W = self.input_resolution # H, W are the patch dimensions, not the pixel dimensions
            
            # To match the dimension for window_partition function
            img_mask = torch.zeros((1, H, W, 1))
            
            # h_slices and w_slices divide a cyclic-shifted image to 9 regions as shown in the paper
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            # Fill out number for each of 9 divided regions
            cnt = 0
            for row in h_slices:
                for col in w_slices:
                    img_mask[:, row, col, :] = cnt
                    cnt += 1
                    
            mask_windows = window_partition(img_mask, self.window_size) # (num_windows, M, M, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            
            # Such a gorgeous code..
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer('attn_mask', attn_mask)
        
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        residual = x # Residual
        x = self.norm1(x)
        x = x.view(B, H, W, C) # H, W refer to the number of "patches" for width and height, not "pixels"
        
        # Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition Windows
        x_windows = window_partition(x, self.window_size) # (num_windows * B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (num_windows * B, window_size ** 2, C)
        
        # W-MSA / SW-MSA

        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (num_windows * B, window_size * window_size, C)
        
        # Merge Windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_combination(attn_windows, self.window_size, H, W) # (B, H', W', C)
        
        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H*W, C)
        
        # Feed Forward
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        
        return x

# %%
class MultiSwinBlock(nn.Module):
    """ Swin Transformer layer for one stage
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks (depending on Swin Version - T, L, ..)
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate (Default: 0.0)
        attn_drop (float, optional): Attention dropout rate (Default: 0.0)
        norm_layer (nn.Module, optional): Normalization layer (Default: nn.LayerNorm)
        patch_merger (nn.Module | NOne, optional): Patch Merging layer at the end of the layer (Default: None)
    """
    
    def __init__(self, 
                dim: int, 
                input_resolution: int, 
                depth: int, 
                num_heads: int, 
                window_size: int,
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None, 
                drop=0., 
                attn_drop=0.,
                norm_layer=nn.LayerNorm, 
                patch_merger=None, 
        ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build  Swin Transformer Blocks
        self.blocks = nn.ModuleList([
            SingleSwinBlock(
                dim=dim, 
                input_resolution=input_resolution,              
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=0 if (idx % 2 == 0) else window_size // 2,
                mlp_ratio = mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop, 
                attn_drop=attn_drop,
                norm_layer=norm_layer
            )
            for idx in range(depth)
        ])
        
        # Patch Merging Layer
        self.patch_merger = patch_merger(input_resolution, dim=dim, norm_layer=norm_layer) if patch_merger is not None else None

            
    def forward(self, x):
        x: torch.Tensor = self.blocks.forward(x)
        
        if self.patch_merger is not None:
            x = self.patch_merger(x)
        
        return x

# %%
class SwinTransformer(nn.Module):
    """ Swin Transformer
    
    Args:
        img_size (int | tuple(int)): Input image size 
        patch_size (int | tuple(int)): Patch size (Default: 4)
        num_channels (int): Number of input image channels (Default: 3)
        embed_dim (int): Patch embedding dimension (Default: 96)
        depths (list[int]): Depth of each Swin-T layer
        num_heads (list[int]): Number of attention heads in different layers
        window_size (int): Window size (Default: 4)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. (Default: 4)
        qkv_bias (bool): If True, add a learnable bias to query, key, value (Default: True)
        qk_scale (float); Override default qk scale of head_dim ** -0.5 if set. (Default: None)
        drop_rate (float): Dropout rate (Default: 0)
        attn_drop_rate (float): Attention dropout rate (Default: 0)
        norm_layer (nn.Module): Normalization layer (Default: nn.LayerNorm)
        ape (bool): Refers to absolute position embedding. If True, add ape to the patch embedding (Default: False)
        patch_norm (bool): If True, add normalization after patch embedding (Default: True)
    """
    
    def __init__(self, 
                img_size: int, 
                patch_size=4,
                num_channels=3, 
                embed_dim=96,
                depths: list[int]=[2, 2, 6, 2], 
                num_heads: list[int]=[2, 4, 8, 16],
                window_size=4,
                mlp_ratio=4.,
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0., 
                norm_layer=nn.LayerNorm, 
        ):
        super().__init__()
        
        if len(depths) != len(num_heads):
            ValueError(f'Length of depths: {depths} must be equal Length of num_heads: {num_heads}')

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbeddings(
            image_size=img_size, 
            patch_size=patch_size, 
            num_channels=num_channels, 
            embed_dim=embed_dim,
        )

        initial_patch_res = img_size // patch_size
                        
        # build layers
        self.layers = nn.ModuleList()
        for depth, idx in enumerate(depths):
            layer = MultiSwinBlock(
                dim=int(embed_dim * (2 ** idx)), # After each layer, the number of channels  are dou
                input_resolution=(
                    initial_patch_res // (2 ** idx), # After patch-merging layer, patches_resolution(H, W) is halved
                    initial_patch_res // (2 ** idx),    
                ),
                depth=depth,
                num_heads=num_heads[idx],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                patch_merger=PatchMerging if (idx < len(depths) -1) else None, # No patch merging at the last stage
            )
            
            self.layers.append(layer)
            
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
        x = self.patch_embed(x) 
        x = self.layers.forward(x)
        return x
    

# %% [markdown]
# ## Model Test

# %%
model = SwinTransformer()
dummy = torch.randn(10, 3, 256, 256)
model(dummy).shape


