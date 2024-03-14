import torch
from torch import nn
from collections import Counter
import numpy as np


class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5 / self.embedding_dim
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()

    def forward(self, i_indices, j_indices):
        i_embeds = self.embedding(i_indices)
        j_embeds = self.embedding(j_indices)
        out1 = self.linear1(i_embeds).squeeze(1)
        out2 = self.linear2(j_embeds).squeeze(1)
        return out1, out2


def create_co_occurrence_matrix(corpus, window_size, vocab_size):
    co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    word_counts = Counter(corpus)
    for i, target_word in enumerate(corpus):
        target_index = i
        context_indices = list(range(max(0, i - window_size), i)) + list(
            range(i + 1, min(len(corpus), i + window_size + 1))
        )
        for context_index in context_indices:
            co_occurrence_matrix[target_word, corpus[context_index]] += 1
            co_occurrence_matrix[corpus[context_index], target_word] += 1
    return co_occurrence_matrix
