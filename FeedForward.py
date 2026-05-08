from torch import nn
from hyperparameters import dropout


class FeedForward(nn.Module):
    """
    Simple linear layer followed by non-linearity layer
    """
    def __init__(self, nEmbd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nEmbd, 4 * nEmbd),
            nn.ReLU(),
            nn.Linear(4 * nEmbd, nEmbd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)