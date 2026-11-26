from typing import Any, Dict, Tuple

import torch
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from lifecaster.lightningmodules.BaseModule import BaseModule
from lifecaster.metrics.precision_at_k import SharedPrecisionAtK
from lifecaster.model.model import PretrainLifecaster
from torchmetrics import MetricCollection


class PretrainModule(BaseModule):
    """Pretraining functionality for the BaseModule"""

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
        # Optimizer
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        warmup_steps,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seqlen=max_seqlen,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
        self.model = PretrainLifecaster(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seqlen=max_seqlen,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
        )

        # FlashAttention version of CELoss (more efficient)
        self.criterion = FACrossEntropyLoss()

        # Metrics setup
        self.val_metrics = self.configure_metrics("val")

    def configure_metrics(self, prefix: str):
        """Sets up Precision@[1,10,100]"""
        return MetricCollection(
            {
                f"{prefix}/Precision@1": SharedPrecisionAtK(
                    k=1, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@10": SharedPrecisionAtK(
                    k=10, max_k=100, reduce="mean"
                ),
                f"{prefix}/Precision@100": SharedPrecisionAtK(
                    k=100, max_k=100, reduce="mean"
                ),
            },
        )

    def standard_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard step -- self.model followed by self.criterion"""
        logits, labels = self.model(batch)
        loss = self.criterion(logits, labels)

        return loss, logits, labels

    def training_step(self, batch: Dict[str, Any], batch_idx) -> torch.Tensor:
        loss, logits, labels = self.standard_step(batch)
        n_valid = (labels != -100).sum()

        self.log("train/loss", loss, batch_size=n_valid)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self.standard_step(batch)
        n_valid = (labels != -100).sum()

        self.log("val/loss", loss, sync_dist=True, batch_size=n_valid)
        self.val_metrics.update(logits, labels)
        self.log_dict(
            self.val_metrics,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=n_valid,
        )
        return loss

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=50, temperature=1.0):
        """Autoregressively sample `max_new_tokens` tokens from the full
        next-token distribution (no top-k / top-p). `batch` needs `event`
        (B, T) and `attn_mask` (B, T); both grow by one token each step.
        Returns the full `event` tensor (prompt + generated)."""
        self.eval()
        event = batch["event"]
        attn_mask = batch["attn_mask"]
        for _ in range(max_new_tokens):
            _, x, _ = self.model.forward(
                {"event": event, "attn_mask": attn_mask}, repad=True
            )
            logits = x[:, -1]
            probs = (logits / temperature).softmax(-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            event = torch.cat([event, next_token], dim=1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones_like(next_token, dtype=attn_mask.dtype)], dim=1
            )
        return event
