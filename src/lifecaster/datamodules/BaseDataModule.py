"""Lightning Data Module for all data-related code"""

import shutil
from pathlib import Path
from typing import List, Literal

import lightning as L
import polars as pl
import pyarrow.dataset as ds
import torch
from flash_attn.bert_padding import unpad_input
from torch.utils.data import DataLoader

from lifecaster.collate_fn import Collate
from lifecaster.dataset.dataset import LMDBDataset
from lifecaster.paths import FPATH, check_and_copy_file_or_dir
from lifecaster.pipeline import DataPipeline
from lifecaster.sampler import UnpadSampler
from lifecaster.utils import get_background_length


class BaseDataModule(L.LightningDataModule):
    """Base Lightning Data Module (BaseLDM)"""

    def __init__(
        self,
        # Directories
        output_dir: Path,
        source_dir: Path,
        # Files
        sources: List[str],
        background: str,
        cohorts: str,
        # Data args
        num_workers,
        n_tokens,
        max_seqlen,
        # Pipeline args
        pipeline_cfg: dict,
    ):
        super().__init__()
        # Init path and data stuff
        self.output_dir = output_dir
        self.source_dir = source_dir
        self.sources = sources
        self.background = background
        self.cohorts = cohorts
        # Init other arg-related stuff
        self.num_workers = num_workers
        self.n_tokens = n_tokens
        self.max_seqlen = max_seqlen

        # Check and copy files between dirs
        output_dir.mkdir(parents=True, exist_ok=True)
        check_and_copy_file_or_dir(output_dir, verbosity=2)
        self.data_pipeline = DataPipeline(
            output_dir=output_dir, source_dir=source_dir, **pipeline_cfg
        )

        self._prepared_data = False

        # Avoid lint complaints
        self.splits = {}
        self.background_length = None
        self.dataset = None
        self.train_dataset, self.val_dataset = None, None
        self.predict_dataset, self.test_dataset = None, None

    def prepare_data(self):
        """Not on all workers"""
        if self._prepared_data:
            return

        self._prepared_data = True
        sources = self.load_sources()
        background = self.load_background()
        cohorts = self.load_cohorts()

        # Reuses if exists otherwise creates
        features_df = self._run_pipeline(sources, background)

        # Reuses if exists or creates
        self.dataset = self._create_dataset(
            features_df, self.output_dir / "dataset.lmdb"
        )

        # Saves back to network drive
        if not FPATH.swap_drives(self.output_dir).exists():
            FPATH.copy_to_opposite_drive(self.output_dir)

        for key, cohort in cohorts.items():
            self.splits[key] = self.prep_split(cohort)

    def load_sources(self):
        source_paths = [
            (self.source_dir / path).with_suffix(".parquet") for path in self.sources
        ]
        for s in source_paths:
            if check_and_copy_file_or_dir(s, verbosity=2) is None:
                raise ValueError(f"Source ({s.name}) not found")
        sources = [ds.dataset(s, format="parquet") for s in source_paths]
        return sources

    def load_background(self):
        background_path = (self.source_dir / self.background).with_suffix(".parquet")
        if check_and_copy_file_or_dir(background_path, verbosity=2) is None:
            raise ValueError(f"Background ({background_path.name}) not found")
        background = pl.read_parquet(background_path)
        self.background_length = get_background_length(background)
        return background

    def load_cohorts(self):
        cohort_paths = {
            key: (self.source_dir / cohort).with_suffix(".parquet")
            for key, cohort in self.cohorts.items()
        }
        for s in cohort_paths.values():
            if check_and_copy_file_or_dir(s, verbosity=2) is None:
                raise ValueError(f"Cohort ({s.name}) not found")
        cohorts = {
            key: pl.read_parquet(path, columns=["person_id"])
            for key, path in cohort_paths.items()
        }
        return cohorts

    def _run_pipeline(self, sources, background):
        # Check if we've already done this stuff
        if (
            tokenized_event := self.data_pipeline.check_already_done(
                local_path=self.output_dir / "tokenized.parquet",
                network_path=self.source_dir / f"{self.output_dir.name}.parquet",
                pprint=False,
            )
        ) is None:
            tokenized_event = self.data_pipeline(sources, background)

        # In case of already done, we still want to load in vocab
        if self.data_pipeline.vocab is None:
            self.data_pipeline.get_vocab(self.sources)

        return tokenized_event

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return LMDBDataset(dataset, path)

    def prep_split(self, cohort):
        """Prepares the split via. a cohort"""
        return cohort.to_dict(as_series=False)

    def setup(self, stage: Literal["fit"] = None):
        """Defaults random splitting"""
        if self.train_dataset is not None:  # Already done the setup
            return
        print("Info: Using predefined splits", self.splits.keys())
        if "train" in self.splits:
            self.train_dataset = self.dataset.subset(self.splits["train"])
        if "val" in self.splits:
            self.val_dataset = self.dataset.subset(self.splits["val"])
        if "test" in self.splits:
            self.predict_dataset = self.dataset.subset(self.splits["test"])
            self.test_dataset = self.dataset.subset(self.splits["test"])

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.max_seqlen,
            self.background_length,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None, shard=False):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        if self.num_workers == 0:
            raise ValueError(
                "Dataset does not work with only a single worker, set it to >0"
            )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn(),
            sampler=self.get_sampler(dataset, sampler, shard),
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )

    def get_sampler(self, dataset, sampler=None, shard=False):
        """Returns a UnpadSampler"""
        lengths = dataset.get_lengths()
        lengths = [lengths[pnr] for pnr in dataset.observations["person_id"]]

        return UnpadSampler(
            lengths,
            n_tokens=self.n_tokens,
            max_seqlen=self.max_seqlen,
            sampler=sampler,
            shard=shard,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        """Initializes the dataset"""
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.init_db()

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        assert self.train_dataset is not None
        return self.get_dataloader(self.train_dataset, shard=True)

    def val_dataloader(self):
        """Returns the val dataloader for self.val_dataset"""
        assert self.val_dataset is not None
        return self.get_dataloader(self.val_dataset, shard=True)

    def predict_dataloader(self):
        """Returns the prediction dataloader for self.predict_dataset"""
        assert self.predict_dataset is not None
        return self.get_dataloader(self.predict_dataset)

    def test_dataloader(self):
        """Returns the test dataloader for self.test_dataset"""
        assert self.test_dataset is not None
        return self.get_dataloader(self.test_dataset)

    def teardown(self, stage: str = None):
        """Copies all contents from output_dir to opposite drive if they do not exist."""
        swapped_path = FPATH.swap_drives(self.output_dir)
        swapped_path.mkdir(parents=True, exist_ok=True)

        for item in self.output_dir.iterdir():
            dest = swapped_path / item.name
            if item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest)
            else:
                if not dest.exists():
                    shutil.copy2(item, dest)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, v in batch.items():
            if torch.is_tensor(v):
                batch[key] = v.to(device, non_blocking=True)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return self.unpad(batch)

    def unpad(self, batch):
        # Unpad inputs
        _, indices, cu_seqlens, max_seqlen_in_batch, total = unpad_input(
            batch["event"].unsqueeze(-1), batch["attn_mask"]
        )
        batch["indices"] = indices
        batch["max_seqlen_in_batch"] = max_seqlen_in_batch
        batch["cu_seqlens"] = cu_seqlens
        batch["total"] = total.sum().item()

        # Flatten inputs
        batch["event"] = batch["event"].flatten()[batch["indices"]]
        batch["abspos"] = batch["abspos"].flatten()[batch["indices"]]
        batch["age"] = batch["age"].flatten()[batch["indices"]]
        batch["segment"] = batch["segment"].flatten()[batch["indices"]]
        return batch
