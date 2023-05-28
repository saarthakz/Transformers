 
# Tiny Shakespeare Dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import torch
import torch.nn as nn
from torch.nn import functional

cuda = torch.cuda.is_available()
print(cuda, torch.cuda.get_device_name())


batch_size = 16
context = 32
num_emb = 128
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
    # print(f"ix is: {ix}")
    x = list(range(len(ix)))
    y = list(range(len(ix)))
    # print(f"Data is: {data[0]}")
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
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(num_emb, head_size, bias=False)    # What the current token contains
        self.query = nn.Linear(num_emb, head_size, bias=False)  # What the current token is looking for
        self.value = nn.Linear(num_emb, head_size, bias=False)  # Additional value addition to input "x"
        self.register_buffer('tril', torch.tril(torch.ones(context, context)))

    def forward(self, x: torch.Tensor):
        
        # We can think of "wei" matrix as the affinity matrix of input tokens, taken across different context lengths

        # Earlier "wei" was same for all the tokens
        # In Self Attention, the "wei" is data dependent

        B, T, C = x.shape
        key_for_x =  self.key(x)   # (B, T, 16)
        query_for_x = self.query(x) # (B, T, 16)
        wei_for_x=  query_for_x @ key_for_x.transpose(-2, -1) * self.head_size **-0.5 # (B, T, 16) @ (B, 16, T) => (B, T, T)
        # compute attention scores ("affinities")
        wei_for_x = wei_for_x.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei_for_x = functional.softmax(wei_for_x, dim=-1) # (B, T, T)

        # Perform the weighted aggregation of the values
        value_for_x = self.value(x) # (B,T,C)
        out = wei_for_x @ value_for_x # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# Multiple Self Attention Heads in Parallel
class MultiHeadAttention(nn.Module):
    

    def __init__(self, num_heads, head_size):
        super().__init__()

        heads = list(range(num_heads))

        for idx in range(num_heads):
            heads[idx] = Head(head_size=head_size)

        self.heads = nn.ModuleList(heads)
        self.proj = nn.Linear(num_emb, num_emb)

    def forward(self, x):
        outputs = list(range(len(self.heads)))
        for idx, head in enumerate(self.heads):
            outputs[idx] = head(x)

        # Outputs are concatenated on the embedding dimension or channel dimension
        out = torch.cat(tensors=outputs, dim=-1)
        return out
    
# A Simple Linear Layer with ReLU
class FeedFoward(nn.Module):

    def __init__(self, num_emb):
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(num_emb, 4 * num_emb), 
            nn.ReLU(),
            nn.Linear(4 * num_emb, num_emb),
        )

    def forward(self, x):
        return self.net(x)

# Transformer Block: Communication followed by Computation 
class Block(nn.Module):

    def __init__(self, num_emb, num_heads):
        # num_emb: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        head_size = num_emb // num_heads
        
        # Self attention heads (multiple in parallel) are used for communication 
        self.self_att = MultiHeadAttention(num_heads, head_size)

        # After communication information is acquired, computation takes place using the Feed Forward Layer
        self.feed_fwd = FeedFoward(num_emb)

    def forward(self, x):
    
        x = x +  self.self_att(x)
        x = x + self.feed_fwd(x)
        return x

 
class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
       
        # Token embedding table is used for token identification encoding
        # Position embedding table is used for token position (in reference to the current context) encoding
        self.token_embedding_table = nn.Embedding(vocab_size, num_emb)
        self.position_embedding_table = nn.Embedding(context, num_emb)

        self.blocks = nn.Sequential(
            Block(num_emb=num_emb, num_heads=4),
            Block(num_emb=num_emb, num_heads=4),
            Block(num_emb=num_emb, num_heads=4),
        )

        # self.ln_f = nn.LayerNorm(num_emb) # Final layer norm
        
        # Language model head used for output
        self.lm_head = nn.Linear(num_emb, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device="cuda")) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        # x = self.ln_f(x) 
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Cross entropy Function: 
            # If logits is an vector, target is a single number
            # If logits is a tensor of shape (count, probabilities), targets is a vector of length "count" 

            # Hence we are resizing the logits and targets 
            # Here we are clubbing all the batches together in both the logits and the targets
            # For each singular target, we have a probability distribution over "dimensionality" number of elements
            # Since "dimensionality" is equal to "vocab_size", the probability is over all the vocab elements
            # Hence the cross entropy calculate loss between the target element and the output element from the probabilities 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop idx to the last context 
            idx_cond = idx[:, -context:]

            # Get the predictions
            logits, loss = self.forward(idx=idx_cond)
            # Focus only on the last  step which contains the output considering the entire context window

            # logits are (batch_size, dimensionality) which is essentially the output vector for each batch
            logits = logits[:, -1, :] 
            # Apply softmax to get probabilities
            probs = functional.softmax(logits, dim=1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence

            # Appended along the context_window hence the context keeps building up
            idx = torch.cat((idx, idx_next), dim=1) # (batch_size, context_window + 1)
        return idx

model = TransformerLanguageModel().to(device=device)

# Print the number of parameters in the model
print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer

# Adaptive Gradient Optimizer
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)

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
open('smore.txt', 'w').write(decode(model.generate(start, max_new_tokens=10000)[0].tolist()))