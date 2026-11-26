import warnings
from typing import Any, Dict

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import LinearLR

from lifecaster.model.model import Lifecaster

warnings.filterwarnings(
    "ignore", message="To copy construct from a tensor", category=UserWarning
)


class BaseModule(pl.LightningModule):
    """Base functionality for the LightningModule"""

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
        # Optimizer (accessed through hparams in subclasses)
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        warmup_steps,
    ):
        super().__init__()
        self.model = Lifecaster(
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

    def configure_optimizers(self):
        """Steps AdamW optimizer and LinearLR scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=torch.tensor(self.hparams.learning_rate),
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
            # fused=True, # Cannot be used with gradient clipping
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LinearLR(
                    optimizer,
                    start_factor=1e-4,
                    end_factor=1,
                    total_iters=self.hparams.warmup_steps,
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def configure_metrics(self, prefix: str):
        """Sets up metrics using a `prefix`"""
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step"""
        raise NotImplementedError

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        raise NotImplementedError

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Test step"""
        raise NotImplementedError
