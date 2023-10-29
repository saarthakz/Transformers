import torch
import torch.nn as nn
from torch.nn import functional

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
# Tokenizer functions
class ByteTokenizer:
  def __init__(self, chars: 'list[str]') -> None:
    # Map creation
    self.stoi = {}
    self.itos = {}
    for idx, char in enumerate(chars):
      self.stoi[char] = idx
      self.itos[idx] = char
    pass
  
  def encode(self, text: str):
    output = list(range(len(text)))
    for idx, char in enumerate(text):
      output[idx] = self.stoi[char]

    return output

  def decode(self, arr: list[int]):
    output = list(range(len(arr)))
    for idx in range(len(arr)):
      output[idx] = self.itos[arr[idx]]

    return "".join(output)

# %%
# Single Self Attention Head
class SelfAttentionHead(nn.Module):

    def __init__(self, emb_dims, head_size):
        super().__init__()
        self.head_size = head_size
        # Key, Query and Value weights are (D, H)
        self.key = nn.Linear(emb_dims, head_size, bias=False)
        self.query = nn.Linear(emb_dims, head_size, bias=False)
        self.value = nn.Linear(emb_dims, head_size, bias=False)


    # Input is (B, C, D) ; Output is (B, C, H)
    def forward(self, key_input: torch.Tensor, query_input: torch.Tensor, value_input: torch.Tensor, mask: torch.Tensor):
        B, C, D = key_input.shape # Batch, Context, Dimensionality
        key: torch.Tensor =  self.key(key_input) # (B, C, D) @ (B, D, H) -> (B, C, H)
        query: torch.Tensor = self.query(query_input) # (B, C, D) @ (B, D, H) -> (B, C, H)
        value: torch.Tensor  = self.value(value_input) # (B, C, D) @ (B, D, H) -> (B, C, H)
        wei: torch.Tensor =  query @ key.transpose(-2, -1) * self.head_size **-0.5 # (B, C, H) @ (B, H, C) => (B, C, C)
        # compute attention scores ("affinities")

        wei = wei.masked_fill(mask, float('-inf')) # (B, C, C)
        wei = functional.softmax(wei, dim=-1) # (B, C, C)

        # perform the weighted aggregation of the values

        out = wei @ value # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out


# %%
# Multiple Self Attention Heads in Parallel
class MultiHeadAttention(nn.Module):
    

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()

        # Key, Query and Value weights are (D, H)
        self.num_heads = num_heads
        self.head_size = head_size
        self.emb_dim = num_heads * head_size # Dimensionality 
        self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.key =   nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.value_proj = nn.Linear(self.emb_dim, self.emb_dim) # Additional layer for inter head communication

    def split(self, x:torch.Tensor):
        B, C, D = x.shape
        x = x.view(B, C, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3) # B, N, C, H

    def forward(self, query_input: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor,mask = None, output_attention = False):

        B, C, D = value_input.shape

        if mask is None:
            # Default mask is the encoder mask
            mask = torch.zeros(size = (C, C)).bool().to(device=value_input.device)

        # B, N, C, H in size
        query= self.split(self.query(query_input))
        key = self.split(self.key(key_input))
        value = self.split(self.value(value_input))

        wei = query @ key.transpose(-2, -1) * self.head_size ** -0.5 # (B, N, C, H) @ (B, N, H, C) => (B, N, C, C)

        wei = wei.masked_fill(mask, float('-inf')) # (B, N, C, C)
        wei = functional.softmax(wei, dim=-1) # (B, N, C, C)
        values = wei @ value # (B, N, C, C) @ (B, N, C, H) -> (B, N, C, H)
        values = values.permute(0, 2, 1, 3) # (B, C, N, H)
        values = values.reshape(B, C, self.emb_dim)
        values = self.value_proj(values)

        if output_attention:
            return values, wei

        return values


# %%
# A Simple Linear Layer with GELU for adding computational abilities
class FeedFoward(nn.Module):

    def __init__(self, emb_dims, dropout = 0):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(emb_dims, 4 * emb_dims), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dims, emb_dims),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# %%
# Transformer Block: Communication followed by Computation 
class Block(nn.Module):

    def __init__(self, emb_dims, num_heads, dropout = 0):
        # emb_dims: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()

        # Divide the embedding dimensions by the number of heads to get the head size
        head_size = emb_dims // num_heads

        # Communication
        self.self_att = MultiHeadAttention(num_heads, head_size)

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


# %%
class Transformer(nn.Module):

    def __init__(self, context, emb_dims, vocab_size, form = "decoder"):
        super().__init__()
       
       
        if form == 'decoder':
            self.mask = torch.triu(torch.full(size = (context,context), fill_value= -torch.inf), diagonal=1).bool().to(device=device)
        else:
            self.mask = torch.zeros(size = (context, context)).bool().to(device=device)
        self.context = context
        self.emb_dims = emb_dims
        self.vocab_size = vocab_size

        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dims)
        self.position_embedding_table = nn.Embedding(context, emb_dims)

        self.blocks = nn.ModuleList([
            Block(emb_dims=emb_dims, num_heads=4),
            Block(emb_dims=emb_dims, num_heads=4),
            Block(emb_dims=emb_dims, num_heads=4),
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(emb_dims) 
        
        # Language model head used for output
        self.lm_head = nn.Linear(emb_dims, vocab_size)

    def forward(self, x, targets=None):
        B, C = x.shape

        # x and targets are both (B,C) tensor of integers
        tok_emb = self.token_embedding_table(x) # (B,C,D)

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(torch.arange(C, device="cuda")) # (C,D)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, self.mask)
        x = self.ln_f(x) 
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, C, D = logits.shape
            logits = logits.view(B*C, D)
            targets = targets.view(B*C)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, C) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last context 
            idx_cond = idx[:, -self.context:]

            # Get the predictions
            logits, loss = self.forward(x=idx_cond)

            # Focus only on the last step which contains the output considering the entire context window
            # logits are (batch_size, context = full context considered time step, dimensionality) which is essentially the output vector for each batch
            logits = logits[:, -1, :] 

            # Apply softmax to get probabilities
            probs = functional.softmax(logits, dim=1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Appended along the context_window hence the context keeps building up
            idx = torch.cat((idx, idx_next), dim=1) # (batch_size, context_window + 1)
        return idx

