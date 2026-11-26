"""
Minimal nanoencoder from GPT-2 with updated architectures (FlashAttention and varlen unpadding).

GPT-style transformer with three concrete heads:
    Lifecaster              - Shared backbone (embedding + transformer stack)
    PretrainLifecaster      - Causal LM pretraining
    FinetuneLifecaster      - For binary classification using prediction tokens
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from flash_attn.bert_padding import pad_input
from lifecaster.model.components.embeddings import LifecasterEmbeddings
from lifecaster.model.components.layer import FlashLayer

# See https://github.com/pytorch/torchtune/issues/1185
# 'Reduce VRAM usage by reducing fragmentation'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Lifecaster(nn.Module):
    """The backbone for the Lifecaster"""

    def __init__(
        self,
        # Embedding related
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_heads,
        num_layers,
        # Attention / behaviour
        dropout,
        attention_dropout,
        bias,
        causal,
    ):
        super().__init__()
        self.embeddings = LifecasterEmbeddings(
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            hidden_size=hidden_size,
        )
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                FlashLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    max_seqlen=max_seqlen,
                    attention_dropout=attention_dropout,
                    bias=bias,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(hidden_size, bias=bias)

        self.apply(self._init_weights)

    @torch.compile(dynamic=True)
    def forward(self, batch: Dict[str, Any], repad: bool = True):
        """Forward pass with unpadded input shape"""
        x = self.embeddings(
            event=batch["event"],
            age=batch["age"],
            abspos=batch["abspos"],
            segment=batch["segment"],
        )

        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, cu_seqlens=batch["cu_seqlens"])
        x = self.layernorm(x)

        # Repad packed (total, hidden_size) tensor back to (B, T, hidden_size).
        if (x.ndim == 2) and repad:
            B, T = batch["attn_mask"].shape
            x = pad_input(x, batch["indices"], B, T)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


class PretrainLifecaster(Lifecaster):
    """Pretraining version of Lifecaster"""

    def __init__(
        self,
        # Embedding related
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_heads,
        num_layers,
        # Attention / behaviour
        dropout,
        attention_dropout,
        bias,
        causal,
    ):
        super().__init__(
            # Embedding related
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            # Model dimensions
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            # Attention / behaviour
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
        )

        self.decoder_pretrain = nn.Linear(hidden_size, vocab_size, bias=bias)
        # Weight-tying (https://paperswithcode.com/method/weight-tying)
        self.decoder_pretrain.weight = self.embeddings.event_embeddings.weight

    def forward(
        self, batch: Dict[str, Any], repad=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward in packed (unpadded) layout and returns unpadded & valid (!= -100 targets) version"""
        x = super().forward(batch, repad=repad)

        # Only score positions whose label is not -100 (sparse prediction)
        labels = batch["target"]
        mask_tokens = labels != -100
        x = x[mask_tokens]
        labels = labels[mask_tokens]

        logits = self.decoder_pretrain(x)

        return logits, labels  # Returns in unpadded (flat) format


class FinetuneLifecaster(Lifecaster):
    """Finetuning version of Lifecaster"""

    def __init__(
        self,
        # Embedding related
        vocab_size,
        max_seqlen,
        # Model dimensions
        hidden_size,
        num_heads,
        num_layers,
        # Attention / behaviour
        dropout,
        attention_dropout,
        bias,
        causal,
        # Finetuning specific
        prediction_windows,
    ):
        super().__init__(
            # Embedding related
            vocab_size=vocab_size,
            max_seqlen=max_seqlen,
            # Model dimensions
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            # Attention / behaviour
            dropout=dropout,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
        )

        self.decoder_finetune = torch.nn.Linear(hidden_size, len(prediction_windows))

    def forward(
        self, batch: Dict[str, Any], repad=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward in packed (unpadded) layout and returns padded version"""
        x = super().forward(batch, repad=repad)  # (bs, slen, dim)

        predict_mask = batch["predict_mask"]
        logits_flat = self.decoder_finetune(x[predict_mask])

        # Reshape logits_flat into target.shape
        logits = self._rebuild_logits_to_target_shape(
            logits_flat, predict_mask, batch["target"].shape
        )

        return logits, batch["target"]  # returns in padded format

    @staticmethod
    def _rebuild_logits_to_target_shape(logits_flat, predict_mask, target_shape):
        b_idx = predict_mask.nonzero(as_tuple=True)[0]  # which person
        within = predict_mask.cumsum(1)[predict_mask] - 1  # which slot for that person
        bs, num_targets, _ = target_shape
        logits = logits_flat.new_full(
            (bs, num_targets, logits_flat.size(-1)), fill_value=torch.nan
        )
        logits[b_idx, within] = logits_flat

        return logits
