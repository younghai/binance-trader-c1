import os
from os.path import pathsep
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def load_text(path):
    with open(path, "r") as f:
        text = f.read().splitlines()

    return text


def to_parquet(df, path, compression="zstd"):
    pq.write_table(table=pa.Table.from_pandas(df), where=path, compression=compression)


def get_filename_by_path(path):
    return Path(path).stem.split(".")[0]


def get_parent_dir(path):
    return Path(path).parent


def to_abs_path(file, relative_path):
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(file)), relative_path)
    )
