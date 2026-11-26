import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

from lifecaster.model.components.rope import FlashRotaryEmbedding


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention using FlashAttention's attention and rotary embedding"""

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
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.max_seqlen = max_seqlen
        # key, query, value projections for all heads, but in a batch (more optimized)
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.rotary_embedding = FlashRotaryEmbedding(self.head_dim)
        self.self_attn = FlashSelfAttention(
            causal=causal, attention_dropout=attention_dropout
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor):
        """Packed (unpadded) rotary_embedding and attention"""
        total, C = x.shape

        # QKV computation
        qkv = self.Wqkv(x)
        qkv = qkv.view(total, 3, self.num_heads, self.head_dim)

        # Applies FlashAttention's RotaryEmbedding
        qkv = self.rotary_embedding(
            qkv, cu_seqlens=cu_seqlens, max_seqlen=self.max_seqlen
        )

        y = self._attend(qkv, cu_seqlens)

        # Reshapes to input shape
        y = y.view(total, C)

        # output projection
        y = self.out_proj(y)

        return y

    @torch.compiler.disable
    def _attend(self, qkv, cu_seqlens):
        # Applies FlashSelfAttention
        return self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=self.max_seqlen)
