import os
import json
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path


class Position:
    def __init__(
        self,
        asset,
        side,
        qty,
        entry_price,
        entry_at,
        n_updated=0,
        profit=None,
        is_exited=False,
    ):
        self.asset = asset
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_at = entry_at
        self.n_updated = n_updated
        self.profit = profit
        self.is_exited = is_exited

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"Position(asset={self.asset}, side={self.side}, qty={self.qty:.3f}, entry_price={self.entry_price:.2f})"

    def __str__(self):
        return self.__repr__()


def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def load_text(path):
    with open(path, "r") as f:
        text = f.read().splitlines()

    return text


def load_json(path):
    with open(path, "r") as f:
        loaded = json.load(f)

    return loaded


def to_parquet(df, path, compression="zstd"):
    pq.write_table(table=pa.Table.from_pandas(df), where=path, compression=compression)


def get_filename_by_path(path):
    return Path(path).stem.split(".")[0]


def to_abs_path(file, relative_path):
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(file)), relative_path)
    )


def initialize_main_logger():
    import logging
    import sys

    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level="INFO",
        handlers=[handler],
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def initialize_trader_logger():
    from handler import SlackHandler
    import logging
    import sys

    handler = logging.StreamHandler(sys.stdout)
    slack_handler = SlackHandler()

    logging.basicConfig(
        level="INFO",
        handlers=[handler, slack_handler],
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
