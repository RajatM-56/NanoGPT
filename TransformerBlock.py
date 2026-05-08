from torch import nn
from AttentionHead import MultiHeadAttention
from FeedForward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block: communication followed by computation
    """

    def __init__(self, nEmbd, numberOfHeads):
        super().__init__()
        headSize = nEmbd // numberOfHeads
        self.selfAttention = MultiHeadAttention(numberOfHeads, headSize)
        self.feedForward = FeedForward(nEmbd)
        self.LayerNorm1 = nn.LayerNorm(nEmbd)
        self.LayerNorm2 = nn.LayerNorm(nEmbd)

    def forward(self, input):
        input = input + self.selfAttention(self.LayerNorm1(input))
        input = input + self.feedForward(self.LayerNorm2(input))
        return input