import torch
from typing import Optional, Callable
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class TimepointMetric(Metric):
    """Computes the Timepoint version of the Metric (indexes pred_times and/or prediction_window)"""

    def __init__(
        self,
        metric_func: Callable,
        pred_time: Optional[int] = None,
        prediction_window: Optional[int] = None,
        ignore_index=-100,
    ):
        super().__init__()
        if pred_time is None:
            pred_time = slice(None)
        self.pred_time = pred_time
        if prediction_window is None:
            prediction_window = slice(None)
        self.prediction_window = prediction_window
        self.ignore_index = ignore_index
        self.metric_func = metric_func

        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits.append(logits)
        self.labels.append(labels.to(torch.int32))

    def compute(self):
        logits = dim_zero_cat(self.logits)[:, self.pred_time, self.prediction_window]
        labels = dim_zero_cat(self.labels)[:, self.pred_time, self.prediction_window]

        mask = (labels != self.ignore_index) & (~labels.isnan())

        x, y = logits[mask], labels[mask]
        if y.numel() == 0 or y.min() == y.max():
            return torch.tensor(torch.nan, device=x.device)
        return self.metric_func(x, y)
