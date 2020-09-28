import pandas as pd
from IPython.display import display, display_markdown


def nan_to_zero(value):
    if str(value) in ("nan", "None"):
        return 0

    return value


def data_loader(path, compression=None):
    if compression is not None:
        return pd.read_csv(path, header=0, index_col=0, compression=compression)

    return pd.read_csv(path, header=0, index_col=0)


def compute_quantile(x, bins):
    if str(x) in ("None", "nan"):
        x = 0

    for idx in range(len(bins) - 1):
        if bins[idx] < x <= bins[idx + 1]:
            return idx

    raise RuntimeError("unreachable")


class Position:
    def __init__(
        self, asset, side, qty, entry_price, entry_at, n_updated=0, is_exited=False
    ):
        self.asset = asset
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_at = entry_at
        self.n_updated = n_updated
        self.is_exited = is_exited

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"Position(asset={self.asset}, side={self.side}, qty={self.qty}, entry_price={self.entry_price:.4f}, n_updated={self.n_updated}, is_exited={str(self.is_exited)})"

    def __str__(self):
        return self.__repr__()
