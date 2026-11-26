"""File for creating various features such as age, abspos, segment, etc."""

from pathlib import Path
from typing import List

import polars as pl
import polars.selectors as cs
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lifecaster.tokenize import tokenize
from lifecaster.utils import calculate_abspos
from tqdm import tqdm


def add_sep_tokens(sep_token: int = 2):
    """Adds seperator tokens to the end of each (TOKENIZED) event"""
    return pl.col("event").list.concat(sep_token)


def create_abspos(
    df: pl.DataFrame, origin_point=pl.datetime(2020, 1, 1, time_unit="ns")
):
    """Creates the abspos columns by doing 'date_col' - origin_point"""
    return df.with_columns(
        abspos=calculate_abspos(pl.col("date_col"), origin_point=origin_point).cast(
            pl.Float32
        )
    )


def create_ages(df: pl.DataFrame, birthdates: pl.DataFrame):
    """Creates ages by joining birthdates and subtracting from date_col"""
    return (
        df.join(birthdates, on="person_id", how="inner")
        .with_columns(
            age=(
                (pl.col("date_col") - pl.col("birthday")).dt.total_days() / 365.25
            ).cast(pl.Float32)
        )
        .drop("birthday")
    )


def create_tokenized_events(
    output_path: Path,
    sources: List[ds.Dataset],
    vocab: dict,
    birthdates: pl.DataFrame,
    sep_token: bool,
    fill_nulls=False,
) -> ds.Dataset:
    """
    Tokenizes and creates events with features, saving it to dataset_name / 'tokenized.parquet'

    Parameters:
        sources: The list of LazyFrames to be processed and tokenized
        vocab: The vocabulary used for tokenization
        birthdates: person_id, birthday dataframe
        dataset_name: The "name" of this data version
        sep_token: decides if each event has an appended [SEP] token
        fill_nulls: replace nulls with [UNK]
    Assumptions:
        Sources to have ID='person_id' and timestamp='date_col'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("person_id", pa.int64()),
            pa.field("event", pa.large_list(pa.int32())),
            pa.field("age", pa.large_list(pa.float32())),
            pa.field("abspos", pa.large_list(pa.float32())),
        ]
    )
    writer = pq.ParquetWriter(output_path, schema=schema)

    for source in tqdm(sources, "Tokenizing sources"):
        for source_batch in source.to_batches():
            # Sometimes there aren't any rows due to filtering of batches
            if source_batch.num_rows == 0:
                continue

            chunk_df = pl.from_arrow(source_batch)
            # Tokenize string columns
            chunk_df = tokenize(chunk_df, vocab)

            # Select event columns (by negation of non-event columns)
            event_columns = ~cs.by_name("person_id", "date_col")

            if fill_nulls:
                chunk_df = chunk_df.with_columns(
                    event_columns.fill_null(vocab["[UNK]"])
                )

            # Convert to dataframe of (person_id, date_col, event)
            chunk_df = (
                chunk_df.with_columns(
                    pl.Series("event", chunk_df.select(event_columns).to_numpy())
                )
                .select("person_id", "date_col", "event")  # Only select needed columns
                .cast(
                    {
                        "event": pl.List(pl.Int32),
                    }
                )  # Converts numpy NaNs to Null
            )
            # Drop Nulls from event list (done here so it's element-wise)
            if not fill_nulls:  # Can safely skip if nulls are filled
                chunk_df = chunk_df.with_columns(
                    pl.col("event").list.drop_nulls()
                ).filter(pl.col("event").list.len() > 0)

            # Create features
            chunk_df = create_ages(chunk_df, birthdates)
            chunk_df = create_abspos(chunk_df)
            # !! Segments is created during collate as it changes on censoring/truncation

            if sep_token:
                chunk_df = chunk_df.with_columns(add_sep_tokens(vocab["[SEP]"]))

            chunk_df = chunk_df.drop("date_col")
            # Duplicate ages and abspos to match event lengths
            chunk_df = chunk_df.with_columns(
                age=pl.col("age").repeat_by(pl.col("event").list.len()),
                abspos=pl.col("abspos").repeat_by(pl.col("event").list.len()),
            )

            writer.write_table(chunk_df.to_arrow())
    writer.close()

    return ds.dataset(output_path)
