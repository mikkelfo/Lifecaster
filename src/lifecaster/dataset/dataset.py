"""Implements the LMDB dataset"""

import gc
import random
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, Union

import lmdb
import numpy as np
import polars as pl
import pyarrow.dataset as ds
import zstandard as zstd
from torch.utils.data import Dataset

from lifecaster.chunking import yield_chunks
from lifecaster.dataset.serializer import Serializer
from lifecaster.paths import check_and_copy_file_or_dir

# Compression/decompression
cctx = zstd.ZstdCompressor(level=9)
dctx = zstd.ZstdDecompressor()


class LMDBDataset(Dataset):
    """LMDB implementation for out-of-core datasets"""

    def __init__(
        self,
        data: ds.Dataset,
        lmdb_path: Path,
        observations: Optional[Dict] = None,
        verbosity=3,
    ):
        self.lmdb_path = lmdb_path
        self.data = data
        self.max_dbs = 4

        check_and_copy_file_or_dir(self.lmdb_path.parent, verbosity=verbosity)

        if not lmdb_path.exists():
            print("Creating", lmdb_path.stem)
            self._create_db(data, str(self.lmdb_path))

        if observations is None:
            observations = self._get_pop()
        self.observations = observations

        self.env = None
        self.data_db = None
        self.event_lengths_db = None
        # The GC is running amok, so we increase the thresholds (default 700, 10, 10) by 10x to achieve a high speedup and useless num_workers in the DataLoader
        gc.set_threshold(700 * 10, 10 * 10, 10 * 10)

    def _estimate_map_size(
        self,
        data: ds.Dataset,
        buffer: int = 0.3,
        sample_size=100_000,
        min_size=5e9,  # 5 GB
    ) -> int:
        """
        Estimate the required map size for the LMDB database.

        Args:
            data (ds.Dataset): The Dataset containing the data.
            buffer (int): The buffer to add to the map_size estimation

        Returns:
            int: The estimated map size in bytes.
        """
        num_rows = data.count_rows()
        sample_idxs = random.choices(range(num_rows), k=sample_size)
        chunk = data.take(sample_idxs).to_pydict()

        # Estimate size of each row
        total_size = 0
        for i, pnr in enumerate(chunk["person_id"]):
            person = {key: [val[i]] for key, val in chunk.items() if key != "person_id"}
            key = str(pnr).encode()
            value = self.encode_compress(person)
            total_size += len(key) + len(value)

        # Estimate map size
        # (total size divided by amount of rows in sample) multiplied by total amount of rows multiplied by (1 + buffer)
        estimated_map_size = int((total_size / (i + 1)) * num_rows * (1 + buffer))
        # Have min sized buffer
        estimated_map_size = max(estimated_map_size, min_size)

        return estimated_map_size

    def _create_db(self, data: ds.Dataset, lmdb_path: Path, map_size=None):
        if map_size is None:
            map_size = self._estimate_map_size(data)

        pop_info = {"seq_lens": [], "pnrs": []}
        with lmdb.open(str(lmdb_path), map_size=map_size, max_dbs=self.max_dbs) as env:
            data_db = env.open_db(b"data")
            event_lengths_db = env.open_db(b"event_lengths")
            pop_info_db = env.open_db(b"pop_info")
            with env.begin(write=True) as txn:
                for chunk_df in yield_chunks(data):
                    for person in (
                        chunk_df.group_by("person_id")
                        .agg(pl.all().sort_by("abspos"))
                        .iter_rows(named=True)
                    ):
                        pnr = person.pop("person_id")
                        key = str(pnr).encode()
                        txn.put(key, self.encode_compress(person), db=data_db)

                        event_lengths = [len(event) for event in person["event"]]
                        txn.put(
                            key,
                            Serializer.encode_lengths(event_lengths),
                            db=event_lengths_db,
                        )

                        pop_info["seq_lens"].append(sum(event_lengths))
                        pop_info["pnrs"].append(pnr)

                txn.put(
                    b"seq_lens",
                    Serializer.encode(pop_info["seq_lens"]),
                    db=pop_info_db,
                )
                txn.put(
                    b"pnrs",
                    Serializer.encode(pop_info["pnrs"], dtype=np.int64),
                    db=pop_info_db,
                )

    def init_db(self):
        """Initialises the self.env and databases required in __getitem__"""
        if self.env is None:
            self.env = self._get_readonly_env()
            self.data_db = self.env.open_db(b"data", create=False)
            self.event_lengths_db = self.env.open_db(b"event_lengths", create=False)

    def _get_readonly_env(self):
        return lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            readahead=False,
            meminit=False,
            create=False,
            max_dbs=self.max_dbs,
        )

    def _get_pop(self):
        with self._get_readonly_env() as env:
            pop_info_db = env.open_db(b"pop_info", create=False)
            with env.begin(buffers=True, db=pop_info_db) as txn:
                observations = Serializer.decode(txn.get(b"pnrs"), dtype=np.int64)
        return {"person_id": observations}

    def __len__(self) -> int:
        return len(self.observations["person_id"])

    def __getitem__(self, idx: Union[int, str, List[int], List[str]]) -> dict:
        with self.env.begin(buffers=True) as txn:
            if isinstance(idx, list) and (
                isinstance(idx[0], str) or isinstance(idx[0], int)
            ):
                return self.__getitems__(idx, txn)
            elif isinstance(idx, int):
                pnr = str(idx)
            elif isinstance(idx, str):
                pnr = idx
            else:
                raise IndexError(f"{type(idx)} is not supported")

            key = pnr.encode()
            output = {
                "data": self.decompress_decode(txn.get(key, db=self.data_db)),
                "event_lengths": Serializer.decode_lengths(
                    txn.get(pnr, db=self.event_lengths_db)
                ),
            }

        return output

    def index_to_pnr(self, idx):
        return str(self.observations["person_id"][idx])

    def __getitems__(
        self, indices: Union[List[str], List[int]], txn: lmdb.Transaction
    ) -> dict:
        pnrs = (
            [self.index_to_pnr(i) for i in indices]
            if isinstance(indices[0], int)
            else indices  # already string
        )
        pnrs = [pnr.encode() for pnr in pnrs]

        with txn.cursor(db=self.data_db) as dcur:
            data = dcur.getmulti(pnrs)
        with txn.cursor(db=self.event_lengths_db) as elcur:
            event_lengths = elcur.getmulti(pnrs)
        decoded_data = [self.decompress_decode(val) for _, val in data]
        decoded_event_lengths = [
            Serializer.decode_lengths(val) for _, val in event_lengths
        ]
        return {"data": decoded_data, "event_lengths": decoded_event_lengths}

    def get_lengths(self):
        """Returns a dict of (pnr, length)"""
        with self._get_readonly_env() as env:
            pop_info_db = env.open_db(b"pop_info", create=False)
            with env.begin(buffers=True, db=pop_info_db) as txn:
                pnrs = Serializer.decode(txn.get(b"pnrs"), dtype=np.int64)
                length = Serializer.decode(txn.get(b"seq_lens"))
        return dict(zip(pnrs, length))

    def split(self, splits: Dict[str, Union[float, List[int]]]) -> Tuple[Self, Self]:
        """If splits is float, use random slitting, if List, split based on lists"""
        first_key = next(iter(splits))
        if isinstance(splits[first_key], float):
            obs = self.split_ratio(self.observations, splits)
        elif isinstance(splits[first_key], list):
            obs = self.split_list(self.observations, splits)
        else:
            raise TypeError(f"{type(first_key)} not supported [float, List[int]]")

        datasets = {}
        for key, subset in obs.items():
            datasets[key] = self.subset(subset)
        return datasets

    @staticmethod
    def split_ratio(data: Dict[str, List], splits: Dict[str, float]):
        """Splits a dictionary using one or more split ratios"""
        assert sum(splits.values()) == 1
        N = len(data["person_id"])
        idxs = list(range(N))

        random.shuffle(idxs)

        last_ratio = 0
        new_splits = {}
        for i, (key, split_ratio) in enumerate(splits.items()):
            start_idx, end_idx = (
                int(N * last_ratio),
                int(N * (last_ratio + split_ratio)),
            )
            if i == (len(splits) - 1):
                end_idx = None

            subset = {
                key: [values[i] for i in idxs[start_idx:end_idx]]
                for key, values in data.items()
            }
            last_ratio += split_ratio
            new_splits[key] = subset

        return new_splits

    @staticmethod
    def split_list(data: Dict[str, List], splits: Dict[str, List[int]]):
        """Splits a dictionary based on Lists of pnrs"""
        first_key = next(iter(splits))
        list_type = type(splits[first_key][0])
        df = pl.DataFrame(data).with_columns(pl.col("person_id").cast(list_type))

        new_splits = {}
        for key, split in splits.items():
            ids_set = set(split)
            subset = df.filter(pl.col("person_id").is_in(ids_set)).to_dict(
                as_series=False
            )
            new_splits[key] = subset
        return new_splits

    def subset(self, observations_subset: dict) -> Self:
        """
        Subset of the dataset based on the observations subset.

        Returns:
            Self: A new Self object with the observations subset.
        """
        return self.__class__(
            self.data, self.lmdb_path, observations_subset, verbosity=2
        )

    def encode_compress(self, person: dict):
        """Serialization followed by zstd level-9 compression"""
        return cctx.compress(Serializer.encode_person(person))

    def decompress_decode(self, buffer):
        """Decompression followed by serialization"""
        return Serializer.decode_person(dctx.decompress(buffer))


class FinetuneLMDBDataset(LMDBDataset):
    """Finetuning version of LMDBDataset, which uses outcomes as observations"""

    def __init__(self, data: ds.Dataset, lmdb_path: Path, outcomes: dict, verbosity=3):
        super().__init__(
            data=data, lmdb_path=lmdb_path, observations=outcomes, verbosity=verbosity
        )

    def __getitem__(self, idx: Union[int, List[int]]):
        output = super().__getitem__(idx)
        if isinstance(idx, int):
            outcome_info = {key: val[idx] for key, val in self.observations.items()}
        elif isinstance(idx, list):
            outcome_info = [
                {key: val[i] for key, val in self.observations.items()} for i in idx
            ]
        else:
            raise IndexError("Only int or List[int] allowed")
        output.update({"outcome_info": outcome_info})
        return output
