import os
import sys
import torch
from torch import nn
import numpy as np

sys.path.append(os.path.abspath("."))
from classes.GloVe import GloVe, create_co_occurrence_matrix

# Sample text corpus
corpus = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]

# Define vocabulary and create word-to-index mapping
vocab = set(corpus)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)

# Hyperparameters
embedding_dim = 50
window_size = 2
learning_rate = 0.01
num_epochs = 100

# Create co-occurrence matrix
co_occurrence_matrix = create_co_occurrence_matrix(corpus, window_size, vocab_size)

# Convert co-occurrence matrix to PyTorch tensors
i_indices, j_indices = np.nonzero(co_occurrence_matrix)
i_indices = torch.LongTensor(i_indices)
j_indices = torch.LongTensor(j_indices)
co_occurrences = torch.FloatTensor(
    co_occurrence_matrix[np.nonzero(co_occurrence_matrix)]
)

# Define GloVe model
model = GloVe(vocab_size, embedding_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output1, output2 = model(i_indices, j_indices)
    loss = criterion(output1, co_occurrences) + criterion(output2, co_occurrences)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Get word embeddings
embeddings = model.embedding.weight.data.numpy()

# Print word embeddings
for i, embedding in enumerate(embeddings):
    print(f"{index_to_word[i]}: {embedding}")
