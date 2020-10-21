import os
import pyarrow.parquet as pq
import pyarrow as pa


def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def load_text(path):
    with open(path, "r") as f:
        text = f.read().splitlines()

    return text


def to_parquet(df, path, compression="zstd"):
    pq.write_table(
        table=pa.Table.from_pandas(df), where=path, compression=compression,
    )
