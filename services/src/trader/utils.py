import pandas as pd
from IPython.display import display, display_markdown


def nan_to_zero(value):
    if str(value) in ("nan", "None"):
        return 0

    return value


def data_loader(path):
    return pd.read_parquet(path)


def compute_quantile(x, bins):
    if str(x) in ("None", "nan"):
        x = 0

    for idx in range(len(bins) - 1):
        if bins[idx] < x <= bins[idx + 1]:
            return idx

    raise RuntimeError("unreachable")
