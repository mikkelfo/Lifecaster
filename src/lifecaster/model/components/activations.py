import torch.nn as nn


class SwiGLU(nn.Module):
    """Implements the SwiGLU activation format (https://arxiv.org/abs/2002.05202)"""

    def __init__(self, dim, bias):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=bias)
        self.l2 = nn.Linear(dim, dim, bias=bias)
        self.l3 = nn.Linear(dim, dim, bias=bias)
        self.swish = nn.SiLU()

    def forward(self, x):
        """Linear(SiLU(Linear_2(x) * Linear_1(x)))"""
        return self.l3(self.swish(self.l2(x)) * self.l1(x))
