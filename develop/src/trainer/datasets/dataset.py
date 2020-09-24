from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable, Optional
import os
import numpy as np
import pandas as pd


FILENAME_TEMPLATE = {"X": "X.csv", "Y": "Y.csv"}


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        lookback_window: int = 60,
        winsorize_threshold: Optional[int] = None,
    ):
        load_files = ["X", "Y"]

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
        assert (self.data_caches["X"].index == self.data_caches["Y"].index).all()

        self.n_data = len(self.data_caches["X"])
        self.lookback_window = lookback_window
        self.winsorize_threshold = winsorize_threshold

    def __len__(self):
        return self.n_data - self.lookback_window + 1

    def __getitem__(self, idx):
        # astype -> Y: int, else: float32
        data_dict = {}
        data_dict["X"] = np.swapaxes(
            self.data_caches["X"]
            .iloc[idx : idx + self.lookback_window]
            .values.astype("float32"),
            0,
            1,
        )
        if self.winsorize_threshold is not None:
            data_dict["X"] = data_dict["X"].clip(
                -self.winsorize_threshold, self.winsorize_threshold
            )

        data_dict["Y"] = (
            self.data_caches["Y"]
            .iloc[idx + self.lookback_window - 1]
            .values.astype("int")
        )

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict

    @property
    def index(self):
        return self.data_caches["X"].index[self.lookback_window - 1 :]
