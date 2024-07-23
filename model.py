
from torch import nn
import torch

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        logits = torch.sum(
            self.context_embedding[x[:,0]] * self.target_embedding[x[:,1]],
            dim=1,
        )
        return torch.sigmoid(logits)