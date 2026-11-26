"""File for tokenization and event creation"""

import json
import shutil
from pathlib import Path
from typing import List

import polars as pl
import pyarrow.dataset as ds

from lifecaster.features import create_tokenized_events
from lifecaster.tokenize import create_vocab
from lifecaster.utils import get_pnrs


class DataPipeline:
    """Class for handling everything related to data processing of the Datamodule"""

    def __init__(
        self,
        output_dir: Path,
        source_dir: Path,
        sep_token: bool,
        fill_nulls: bool = False,
        subset_background: bool = False,
        cutoff: int = 0,
    ):
        self.output_dir = output_dir
        self.source_dir = source_dir

        self.sep_token = sep_token
        self.fill_nulls = fill_nulls
        self.subset_background = subset_background
        self.cutoff = cutoff

        self.vocab = None

    def __call__(
        self,
        sources: List[ds.Dataset],
        background: pl.DataFrame,
    ):
        """Does all data processing required to create features"""
        assert {"person_id", "date_col"}.issubset(background.columns), (
            "Required cols: person_id, date_col"
        )

        # Subset background on sources
        if self.subset_background:
            background = self.get_background_subset(sources, background)
        birthdates = background.select("person_id", birthday="date_col")

        # Prepend background to sources
        if len(background.columns) > 2:
            sources = [ds.dataset(background.to_arrow())] + sources

        # Get vocab
        vocab = self.get_vocab(sources)

        # Get tokenized event Dataset
        tokenized_event = self.get_tokenized_event(sources, vocab, birthdates)

        return tokenized_event

    @staticmethod
    def _load_if_exists(path: Path, backend=None, pprint=True):
        if path.exists():
            if pprint:
                print("Loading", path.stem)
            if path.suffix == ".parquet":
                if backend == "arrow":
                    return ds.dataset(path, format="parquet")
                elif backend == "polars":
                    return pl.read_parquet(path)
                else:
                    raise ValueError(
                        "Only 'arrow' or 'polars' backend supported", backend
                    )
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                raise ValueError("Only .parquet and .json files allowed")
        else:
            if pprint:
                print("Creating", path.stem)
            return None

    def get_background_subset(
        self, sources: List[ds.Dataset], background_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Load or subset background with pnrs of sources"""
        background_subset_path = self.output_dir / "background_subset.parquet"

        # Load or subset background
        if (
            background_subset := self._load_if_exists(
                background_subset_path, backend="polars"
            )
        ) is None:
            sources_pnrs = get_pnrs(sources)
            background_subset = background_df.join(
                pl.DataFrame({"person_id": sources_pnrs.tolist()}), on="person_id"
            )
            background_subset.write_parquet(background_subset_path)
        return background_subset

    def get_vocab(self, sources: List[ds.Dataset]) -> dict:
        """Load or create the vocabulary"""
        vocab_path = self.output_dir / "vocab.json"

        if (vocab := self._load_if_exists(vocab_path)) is None:
            vocab = create_vocab(sources, cutoff=self.cutoff)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f)
        self.vocab = vocab
        return vocab

    def get_tokenized_event(
        self, sources: List[ds.Dataset], vocab: dict, birthdates: pl.DataFrame
    ) -> ds.Dataset:
        """Load or create the tokenized event dataframe"""
        # Tokenize parquet is moved to network to conserve space on local IO
        tokenized_path_local = self.output_dir / "tokenized.parquet"
        tokenized_path_network = self.source_dir / f"{self.output_dir.name}.parquet"

        if (
            tokenized_event := self.check_already_done(
                tokenized_path_local, tokenized_path_network
            )
        ) is not None:
            return tokenized_event
        tokenized_event = create_tokenized_events(
            output_path=tokenized_path_local,
            sources=sources,
            vocab=vocab,
            birthdates=birthdates,
            sep_token=self.sep_token,
            fill_nulls=self.fill_nulls,
        )  # tokenized_event is saved to tokenized_path_local within create_tokenized_events function
        shutil.copy2(
            tokenized_path_local,
            tokenized_path_network,
        )  # Copies it back onto the tokenized_path_network and renames
        return tokenized_event

    def check_already_done(self, local_path, network_path, pprint=True):
        # Load from local io (backwards compatibility)
        if (
            tokenized_event := self._load_if_exists(
                local_path, backend="arrow", pprint=pprint
            )
        ) is not None:
            # In case of already done, we still want to load in vocab
            if self.vocab is None:
                self.get_vocab(None)
            return tokenized_event

        # Load first from network IO
        if (
            tokenized_event := self._load_if_exists(
                network_path, backend="arrow", pprint=pprint
            )
        ) is not None:
            # In case of already done, we still want to load in vocab
            if self.vocab is None:
                self.get_vocab(None)
            return tokenized_event

        return None
