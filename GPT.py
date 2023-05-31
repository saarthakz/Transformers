 
# Tiny Shakespeare Dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional

cuda = torch.cuda.is_available()
print(cuda, torch.cuda.get_device_name())


batch_size = 16
context = 32
emb_dims = 128
eval_iters = 200
eval_interval = 500
device = 'cuda' if cuda else 'cpu'
max_iters = 10000

# Reading the file
file = open('input.txt', 'r', encoding='utf-8')
text = file.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Map creation
stoi = {}
itos = {}
for idx, char in enumerate(chars):
  stoi[char] = idx
  itos[idx] = char


def encode(text: str):
  output = list(range(len(text)))
  for idx, char in enumerate(text):
    output[idx] = stoi[char]

  return output

def decode(arr: list[int]):
  output = list(range(len(arr)))
  for idx in range(len(arr)):
    output[idx] = itos[arr[idx]]

  return "".join(output)

 
# Encoding the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

 
# Split up the data into train and validation sets
split = int(0.9 * len(data)) # First 90% will be training data, rest validation data
train_data = data[:split]
val_data = data[split:]

 
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # ix is a list of random offsets. Length of ix list is equal to the batch_size. It is essentially used to sample "batch_size" number of rows from the data
    ix = torch.randint(len(data) - context, (batch_size,))
    x = list(range(len(ix)))
    y = list(range(len(ix)))

    for idx, offset in enumerate(ix):
      x[idx] = data[offset : offset+context]
      y[idx] = data[offset+1 : offset+context+1]

    # print(len(x), len(y))
    x = torch.stack(x).to(device=device)
    y = torch.stack(y).to(device=device)

    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Single Self Attention Head
class DecoderHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        # Key, Query and Value weights are (D, H)
        self.key = nn.Linear(emb_dims, head_size, bias=False)
        self.query = nn.Linear(emb_dims, head_size, bias=False)
        self.value = nn.Linear(emb_dims, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context, context)))

    # Input is (B, C, D) ; Output is (B, C, H)
    def forward(self, x: torch.Tensor):
        B, C, D = x.shape # Batch, Context, Dimensionality
        key_for_x =  self.key(x) # (B, C, D) @ (B, D, H) -> (B, C, H)
        query_for_x = self.query(x) # (B, C, D) @ (B, D, H) -> (B, C, H)
        value_for_x = self.value(x) # (B, C, D) @ (B, D, H) -> (B, C, H)
        wei_for_x=  query_for_x @ key_for_x.transpose(-2, -1) * self.head_size **-0.5 # (B, C, H) @ (B, H, C) => (B, C, C)
        # compute attention scores ("affinities")

        wei_for_x = wei_for_x.masked_fill(self.tril[:C, :C] == 0, float('-inf')) # (B, C, C)
        wei_for_x = functional.softmax(wei_for_x, dim=-1) # (B, C, C)

        # perform the weighted aggregation of the values

        out = wei_for_x @ value_for_x # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out

# Multiple Self Attention Heads in Parallel
class DecoderMultiHeadAttention(nn.Module):
    

    def __init__(self, num_heads, head_size):
        super().__init__()

        heads = list(range(num_heads))

        for idx in range(num_heads):
            heads[idx] = DecoderHead(head_size=head_size)

        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        outputs = list(range(len(self.heads)))
        for idx, head in enumerate(self.heads):
            outputs[idx] = head(x)

        # Outputs are concatenated on the embedding dimension or channel dimension
        out = torch.cat(tensors=outputs, dim=-1)
        return out
    
# A Simple Linear Layer with ReLU for adding computational abilities
class FeedFoward(nn.Module):

    def __init__(self, emb_dims):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(emb_dims, 4 * emb_dims), 
            nn.ReLU(),
            nn.Linear(4 * emb_dims, emb_dims),
        )

    def forward(self, x):
        return self.net(x)

# Transformer Block: Communication followed by Computation 
class Block(nn.Module):

    def __init__(self, emb_dims, num_heads):
        # emb_dims: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()

        # Divide the embedding dimensions by the number of heads to get the head size
        head_size = emb_dims // num_heads

        # Communication
        self.self_att = DecoderMultiHeadAttention(num_heads, head_size)

        # Computation
        self.feed_fwd = FeedFoward(emb_dims)

        # Adding Layer Normalization
        self.ln1 = nn.LayerNorm(emb_dims)
        self.ln2 = nn.LayerNorm(emb_dims)


    def forward(self, x):
        # Residual connections allow the network to learn the simplest possible function. No matter how many complex layer we start by learning a linear function and the complex layers add in non linearity as needed to learn true function.
        x = x + self.self_att(self.ln1(x))
        x = x + self.feed_fwd(self.ln2(x))
        return x

 
class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()
       
        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dims)
        self.position_embedding_table = nn.Embedding(context, emb_dims)

        self.blocks = nn.Sequential(
            Block(emb_dims=emb_dims, num_heads=4),
            Block(emb_dims=emb_dims, num_heads=4),
            Block(emb_dims=emb_dims, num_heads=4),
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(emb_dims) 
        
        # Language model head used for output
        self.lm_head = nn.Linear(emb_dims, vocab_size)

    def forward(self, idx, targets=None):
        B, C = idx.shape

        # idx and targets are both (B,C) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,C,D)

        # Getting the position embedding for all the positions, starting from 0 -> context - 1
        pos_emb = self.position_embedding_table(torch.arange(C, device="cuda")) # (C,D)
        x = tok_emb + pos_emb
        x = self.blocks(x)
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
            idx_cond = idx[:, -context:]

            # Get the predictions
            logits, loss = self.forward(idx=idx_cond)

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

model = DecoderBlock().to(device=device)

# Print the number of parameters in the model
print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model=model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start = torch.zeros((1, 1), dtype=torch.long, device=device)
open('more.txt', 'w').write(decode(model.generate(start, max_new_tokens=10000)[0].tolist()))