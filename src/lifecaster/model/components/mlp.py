import torch.nn as nn


class MLP(nn.Module):
    """Standard MLP(Linear, GELU, Linear)"""

    def __init__(self, hidden_size, bias):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=bias)

    def forward(self, x):
        """Linear --> GELU --> Linear"""
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y
