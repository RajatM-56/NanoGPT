import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import nEmbd, blockSize, dropout


class AttentionHead(nn.Module):
    """
    One Single attention head
    """

    def __init__(self, headSize):
        super().__init__()
        self.key = nn.Linear(nEmbd, headSize, bias=False)
        self.query = nn.Linear(nEmbd, headSize, bias=False)
        self.value = nn.Linear(nEmbd, headSize, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(blockSize, blockSize)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        attentionScores = k @ q.transpose(-2, -1) * C ** -0.5
        attentionScores = attentionScores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attentionScores = F.softmax(attentionScores, dim=-1)
        attentionScores = self.dropout(attentionScores)

        v = self.value(x)
        out = attentionScores @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Head in parallel
    """

    def __init__(self, numberOfHeads, headSize):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(headSize) for _ in range(numberOfHeads)])
        self.projection = nn.Linear(nEmbd, nEmbd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out