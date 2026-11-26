"""Functions responsible for creating in-memory chunks of big parquet files"""

from pathlib import Path
from typing import List, Union

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from lifecaster.utils import get_pnrs


def yield_chunks(sources: Union[ds.Dataset, List[ds.Dataset]], chunk_size=20_000):
    """Yields chunks (pl.DataFrame) from sources (ds.Dataset)"""
    pnrs = get_pnrs(sources)

    for i in tqdm(range(0, len(pnrs), chunk_size), "Creating Dataset"):
        chunk_pnrs = pnrs[i : i + chunk_size]
        if isinstance(sources, ds.Dataset):
            yield get_chunk(sources, chunk_pnrs)
        elif isinstance(sources, list):
            yield [get_chunk(source, chunk_pnrs) for source in sources]
        else:
            raise ValueError(
                f"Wrong typing {type(sources)}, only ds.Dataset or List[ds.Dataset]"
            )


def get_chunk(source: ds.Dataset, chunk_pnrs: List) -> pl.DataFrame:
    """Subsets source into a table using chunk_pnrs and converts to pl.DataFrame"""
    table = source.to_table(filter=pc.is_in(pc.field("person_id"), chunk_pnrs))
    return pl.from_arrow(table)


def write_dataset_to_parquet_in_batches(
    dataset: ds.Dataset, output_path: Path, batch_size: int = 10_000_000
) -> ds.Dataset:
    """
    Write a PyArrow Dataset to Parquet in batches and return a new Dataset pointing to the output path.
    Faster and less memory than ds.write_dataset()

    Args:
        dataset (ds.Dataset): PyArrow Dataset to write.
        output_path (Path): Path to write the Parquet file.
        batch_size (int): Number of rows per batch.

    Returns:
        ds.Dataset: A new PyArrow Dataset pointing to the output path.
    """
    with pq.ParquetWriter(output_path, dataset.schema) as writer:
        for batch in dataset.to_batches(batch_size=batch_size):
            writer.write_table(pa.Table.from_batches([batch]))

    return ds.dataset(output_path, format="parquet")
