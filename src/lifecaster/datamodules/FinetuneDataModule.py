import math
from pathlib import Path
from typing import Dict, List, Literal

import polars as pl
import pyarrow.dataset as ds
from torch.utils.data import WeightedRandomSampler

from lifecaster.collate_fn import PredictCensorCollate
from lifecaster.datamodules.BaseDataModule import BaseDataModule
from lifecaster.dataset.dataset import FinetuneLMDBDataset
from lifecaster.utils import calculate_abspos, create_weights

ONE_YEAR_ABSPOS = 365.25 * 24


class FinetuneDataModule(BaseDataModule):
    """Lightning Data module for Predict token Finetuning"""

    def __init__(
        self,
        # Paths
        output_dir: Path,
        source_dir: Path,
        # Files
        sources: List[ds.Dataset],
        background: pl.DataFrame,
        cohorts: dict,
        # Data args
        num_workers,
        n_tokens,
        max_seqlen,
        # Pipeline args
        pipeline_cfg: dict,
        # Outcome specific
        outcomes: pl.DataFrame,
        prediction_windows: List[float],
    ):
        super().__init__(
            output_dir=output_dir,
            source_dir=source_dir,
            sources=sources,
            background=background,
            cohorts=cohorts,
            num_workers=num_workers,
            n_tokens=n_tokens,
            max_seqlen=max_seqlen,
            pipeline_cfg=pipeline_cfg,
        )
        self.outcomes = self.prep_outcomes(
            pl.read_parquet((self.source_dir / outcomes).with_suffix(".parquet"))
        )
        self.prediction_windows = prediction_windows

    def prep_split(self, cohort) -> Dict[str, List]:
        """Subsets self.outcomes on cohort"""
        df = cohort.join(self.outcomes, on="person_id")
        df = df.to_dict(as_series=False)
        return df

    def prep_outcomes(self, outcomes):
        """Processes the outcomes"""
        return outcomes.with_columns(
            calculate_abspos(pl.col("censor")),
            calculate_abspos(pl.col("outcome")),
            pl.col("predict").list.eval(calculate_abspos(pl.element())),
        )

    def train_dataloader(self):
        train_outcomes = [
            int(out is not None) for out in self.train_dataset.observations["outcome"]
        ]
        weights = create_weights(train_outcomes, op=math.sqrt)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_outcomes), replacement=True
        )
        return self.get_dataloader(self.train_dataset, sampler=sampler)

    @staticmethod
    def _create_dataset(dataset: ds.Dataset, path: Path):
        return FinetuneLMDBDataset(
            dataset, path, outcomes=None
        )  # outcomes get set in .setup()

    def collate_fn(self):
        return PredictCensorCollate(
            max_seqlen=self.max_seqlen,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
        )
