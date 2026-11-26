""" " Functions to dump data"""

from pathlib import Path
from typing import List, Optional

import oracledb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from lifecaster.paths import FPATH
from sqlalchemy import create_engine
from tqdm import tqdm


def execute_query_to_single_parquet_file(
    database: str,
    sql_script: str,
    filepath: Path,
    schema: Optional[str] = None,
    chunk_size: int = 50_000_000,
) -> None:
    """
    Executes a SQL query on a specified database, saves the results in a single Parquet file, and
    copies it back to the network drive.

    Args:
        database (str): The name of the database to connect to.
        sql_script (str): The SQL script to execute.
        filepath (Path): The path where the Parquet file will be stored. TIP: Use a local drive
        schema (Optional[str]): The schema to use within the database. Defaults to None.
        chunk_size (int): The number of rows per chunk for splitting the SQL output.
            Defaults to 50,000,000.
    """
    oracledb.init_oracle_client()

    # Create connection string and engine
    if schema:
        connection_string = f"oracle+oracledb://[{schema}]@{database}"
    else:
        connection_string = f"oracle+oracledb://{database}"
    engine = create_engine(connection_string)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    # save to parquet in chunks
    schema = None
    with engine.connect() as connection:
        for df in tqdm(pd.read_sql(sql_script, con=connection, chunksize=chunk_size)):
            table = pa.Table.from_pandas(df, schema=schema)
            del df
            if writer is None:
                writer = pq.ParquetWriter(filepath, schema=table.schema)
                schema = table.schema
            writer.write_table(table)
            del table

    writer.close()

    FPATH.copy_to_opposite_drive(filepath)


def replace_zero_with_null(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Replace values with null in specified columns based on their data type:
    - For string columns: replace '0' and '00' with null.
    - For numerical columns: replace 0 with null.

    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        columns (list[str]): List of column names to apply replacements on.

    Returns:
        pl.DataFrame: Updated DataFrame with specified replacements.
    """
    for col in columns:
        # Check if the column is of numeric type (integer or float)
        if pl.Int64 == df.schema[col] or pl.Float64 == df.schema[col]:
            df = df.with_columns(pl.col(col).replace({0: None}))
        # Check if the column is of string type
        elif pl.Utf8 == df.schema[col]:
            df = df.with_columns(pl.col(col).replace(["0", "00"], None))
    return df


def remove_prefix_from_cols(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Removes a specified prefix from all columns that start with it and replaces
    the old columns with the renamed ones in the dataframe.

    Args:
        df (pl.DataFrame): The input Polars dataframe.
        prefix (str): The prefix to remove from column names.

    Returns:
        pl.DataFrame: DataFrame with specified prefix removed from applicable column names.
    """
    return df.rename(
        {col: col[len(prefix) :] for col in df.columns if col.startswith(prefix)}
    )


def bin_numeric_column(
    df: pl.DataFrame, col: str, cutoffs: List[float]
) -> pl.DataFrame:
    """
    Bin a numeric column into string labels based on provided cutoffs.

    For each adjacent pair of cutoffs, values greater than the lower cutoff
    and less than or equal to the upper cutoff are assigned a label
    "bin_{lower}_{upper}". Values above the last cutoff are labeled as
    "over_{last_cutoff}". Values not falling into any bin are labeled as "unbinned".

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        col (str): The name of the column to bin.
        cutoffs (List[float]): A sorted list of cutoff values.

    Returns:
        pl.DataFrame: The DataFrame with an added column "{col}_binned" containing bin labels.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not cutoffs:
        raise ValueError("Cutoffs list cannot be empty.")

    expr = None
    # Build the when-then chain for adjacent cutoff pairs.
    for i in range(len(cutoffs) - 1):
        lower, upper = cutoffs[i], cutoffs[i + 1]
        condition = (pl.col(col) > lower) & (pl.col(col) <= upper)
        if lower + 1 == upper:
            label = f"manual_bin_{upper}"
        else:
            label = f"manual_bin_{lower}_{upper}"
        if expr is None:
            expr = pl.when(condition).then(pl.lit(label))
        else:
            expr = expr.when(condition).then(pl.lit(label))

    # Condition for values above the last cutoff.
    expr = (
        expr.when(pl.col(col) > cutoffs[-1])
        .then(pl.lit(f"over_{cutoffs[-1]}"))
        .otherwise(pl.lit(None, pl.Utf8))
    )

    return df.with_columns(expr.alias(f"manual_bin_{col}"))
