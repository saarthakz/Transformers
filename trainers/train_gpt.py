import os
import sys
sys.path.append(os.path.abspath("."))
from classes.Transformers import ByteTokenizer, Transformer
from utils.logger import Logger
from tqdm.auto import tqdm
from torch.utils.data import Dataset as torchDataset, DataLoader
from torch.nn import functional
import torch.nn as nn
import torch

cuda = torch.cuda.is_available()
print(cuda, torch.cuda.get_device_name())

# %%
batch_size = 128
context = 256
emb_dims = 128
num_heads = 4
print_interval = 500
device = 'cuda' if cuda else 'cpu'
max_iters = 100
epochs = 1
model_dir = './models/gpt'
os.makedirs(model_dir, exist_ok=True)
# %%
# Reading the file
file = open('./data/input.txt', 'r', encoding='utf-8')
text = file.read()

# %%
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# %%
tokenizer = ByteTokenizer(chars)

# %%

class Dataset(torchDataset):
    def __init__(self, text: str) -> None:
        self.data = tokenizer.encode(text)

    def __getitem__(self, index):
        x = self.data[index: index + context]
        y = self.data[index + 1: index + context + 1]

        return torch.tensor(x).to(device), torch.tensor(y).to(device)

    def __len__(self):
        return len(self.data) - context - 1


# %%
dataset = Dataset(text=text)
print(len(dataset))

# %%
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# %%
model = Transformer(context=context, emb_dims=emb_dims,
                    vocab_size=vocab_size, num_heads=num_heads).to(device=device)

# Print the number of parameters in the model
print(sum(param.numel() for param in model.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# %%
# Training loop

progress_bar = tqdm(range(epochs * len(dataloader)))
logger = Logger(os.path.join(model_dir, 'log.txt'))

for epoch in range(epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        # every once in a while evaluate the loss on train and val sets
        if step % print_interval == 0:
            log = f"step {step}: train loss {total_loss / (step + 1)}"
            logger.log(log) 
            tqdm.write(log)

        x, y = batch

        # evaluate the loss
        logits, loss = model.forward(x=x, targets=y)
        total_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        progress_bar.update(1)


torch.save(model.state_dict(), './models/gpt/makemore.pt')
# %%
# Generate data
start = torch.zeros((1, 1), dtype=torch.long, device=device)
open('./data/more.txt', 'w').write(tokenizer.decode(
    model.generate(start, max_new_tokens=10000)[0].tolist()))
