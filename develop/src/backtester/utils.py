import pandas as pd
from IPython.display import display, display_markdown


def data_loader(path, compression=None):
    if compression is not None:
        return pd.read_csv(path, header=0, index_col=0, compression=compression)

    return pd.read_csv(path, header=0, index_col=0)


def compute_quantile(x, bins):
    for idx in range(len(bins) - 1):
        if bins[idx] < x <= bins[idx + 1]:
            return idx

    raise RuntimeError("unreachable")


class Position:
    def __init__(self, asset, side, qty, entry_price, entry_at, base_currency):
        self.asset = asset
        self.side = side
        self.qty = qty
        self.entry_price = entry_price
        self.entry_at = entry_at
        self.base_currency = base_currency

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"Position(asset={self.asset}, side={self.side}, qty={self.qty}, entry_price={self.entry_price:.4f}, base_currency={self.base_currency})"

    def __str__(self):
        return self.__repr__()
