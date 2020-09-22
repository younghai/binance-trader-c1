from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable
import os
import numpy as np
import pandas as pd


FILENAME_TEMPLATE = {"X": "X.csv", "Y": "Y.csv"}


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        load_files: List[str] = ["X", "Y"],
        lookback_window: int = 30,
    ):
        self.data_caches = {
            data_type: pd.read_csv(
                os.path.join(data_dir, FILENAME_TEMPLATE[data_type]),
                header=0,
                index_col=0,
                compression="gzip",
            )
            for data_type in load_files
        }
        self.transforms = transforms

        # Check if all index are same.
        if set(["X", "Y"]).issubset(load_files):
            assert (self.data_caches["X"].index == self.data_caches["Y"].index).all()

        self.n_data = len(self.data_caches["X"])
        self.lookback_window = lookback_window

    def __len__(self):
        return self.n_data - self.lookback_window + 1

    def __getitem__(self, idx):
        # astype -> Y: int, else: float32
        data_dict = {
            data_type: data_cache.iloc[idx + self.lookback_window - 1].values.astype(
                "int"
            )
            if data_type in ("Y")
            else np.swapaxes(
                data_cache.iloc[idx : idx + self.lookback_window].values.astype(
                    "float32"
                ),
                0,
                1,
            )
            for data_type, data_cache in self.data_caches.items()
        }

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict

    @property
    def index(self):
        return self.data_caches["X"].index[self.lookback_window - 1 :]
