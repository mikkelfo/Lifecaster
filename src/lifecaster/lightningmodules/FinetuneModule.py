from typing import Any, Dict, Tuple

import torch
from torchmetrics import MetricCollection
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from lifecaster.lightningmodules.BaseModule import BaseModule
from lifecaster.metrics.timepoint import TimepointMetric
from lifecaster.model.model import FinetuneLifecaster


class FinetuneModule(BaseModule):
    """Finetuning functionality for the LightningModule"""

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
        # Finetuning specific
        pred_times,
        prediction_windows,
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
        self.pred_times = pred_times
        self.prediction_windows = prediction_windows
        self.save_hyperparameters()
        self.model = FinetuneLifecaster(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seqlen=max_seqlen,
            attention_dropout=attention_dropout,
            bias=bias,
            causal=causal,
            prediction_windows=prediction_windows,
        )

        self.val_metrics = self.configure_metrics("val")
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def configure_metrics(self, prefix: str):
        """Sets up 'Timepoint' metrics to compute per pred_time/prediction_window metrics"""
        metrics = {}
        if self.pred_times is not None:
            metrics.update(
                {
                    f"{prefix}/AUROC_time{time}": TimepointMetric(
                        metric_func=binary_auroc, pred_time=i
                    )
                    for i, time in enumerate(self.pred_times)
                }
            )
            metrics.update(
                {
                    f"{prefix}/AUPRC_time{time}": TimepointMetric(
                        metric_func=binary_average_precision, pred_time=i
                    )
                    for i, time in enumerate(self.pred_times)
                }
            )
        if len(self.prediction_windows) > 1:
            metrics.update(
                {
                    f"{prefix}/AUROC_window{window}": TimepointMetric(
                        metric_func=binary_auroc, prediction_window=i
                    )
                    for i, window in enumerate(self.prediction_windows)
                }
            )
            metrics.update(
                {
                    f"{prefix}/AUPRC_window{window}": TimepointMetric(
                        metric_func=binary_average_precision, prediction_window=i
                    )
                    for i, window in enumerate(self.prediction_windows)
                }
            )
        return MetricCollection(metrics)

    def standard_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, labels = self.model(batch)

        target_mask = batch["target"] != -100
        loss = self.criterion(logits[target_mask], labels[target_mask])

        return loss, logits, labels

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self.standard_step(batch)
        n_valid = (labels != -100).sum()

        self.log("train/loss", loss, sync_dist=True, batch_size=n_valid)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self.standard_step(batch)
        n_valid = (labels != -100).sum()

        self.log("val/loss", loss, sync_dist=True, batch_size=n_valid)
        self.val_metrics.update(logits, labels)

        return loss

    def on_validation_epoch_end(self):
        """We have to call this ourselves to compute the MEAN metrics"""
        # Call .compute to get metric vals
        out = self.val_metrics.compute()

        # Calculate the mean metrics
        metric_keys = set([k.split("_")[0] for k in out])
        for key in metric_keys:
            vals = torch.stack([out[k] for k in out.keys() if f"{key}" in k])
            out[f"{key}_mean"] = vals[~vals.isnan()].mean()

        # Have to log and reset ourselves
        self.log_dict(out)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
