"""The backbone layer for the Lifecaster"""

import torch
import torch.nn as nn
from lifecaster.model.components.mha import MultiHeadAttention
from lifecaster.model.components.mlp import MLP


class FlashLayer(nn.Module):
    """Standard layer with pre-norm LayerNorm, MultiHeadAttention and MLP"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seqlen=8192,
        attention_dropout=0.0,
        bias=False,
        causal=True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, bias=bias)
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_seqlen=max_seqlen,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
        )
        self.ln_2 = nn.LayerNorm(hidden_size, bias=bias)
        self.mlp = MLP(hidden_size, bias=bias)

    def forward(self, x, cu_seqlens: torch.Tensor):
        """MHA followed by MLP with pre-norm LayerNorm"""
        x = x + self.attn(self.ln_1(x), cu_seqlens=cu_seqlens).to(
            x.dtype
        )  # Manual upcasting due to issues (unpad + compile)
        x = x + self.mlp(self.ln_2(x)).to(
            x.dtype
        )  # Manual upcasting due to issues (unpad + compile)
        return x
