import torch.nn as nn
import torch.nn.functional as F
from AttentionHead import MultiHeadAttention
from TransformerBlock import TransformerBlock
from hyperparameters import *
from FeedForward import FeedForward

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(charsetSize, nEmbd)
        self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
        self.transformerBlocks = nn.Sequential(
            *[TransformerBlock(nEmbd, numberOfHeads=numberOfHeads) for _ in range(numberOfLayers)]
        )
        self.layerNorm = nn.LayerNorm(nEmbd)

        self.lmHead = nn.Linear(nEmbd, charsetSize)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tokenEmbeddings = self.tokenEmbeddingTable(idx)
        positionalEmbeddings = self.positionEmbeddingTable(torch.arange(T, device=device))
        x = tokenEmbeddings + positionalEmbeddings
        x = self.transformerBlocks(x)
        x = self.layerNorm(x)
        logits = self.lmHead(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, maxNewTokens):
        device = next(self.parameters()).device
        idx = idx.to(device)

        for _ in range(maxNewTokens):
            idx_cond = idx[:, -blockSize:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nextIdx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nextIdx), dim=1)

        return idx
