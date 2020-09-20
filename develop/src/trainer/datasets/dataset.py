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
        assert all(
            [
                (self.data_caches[0].index == data_cache.index).all()
                for data_cache in self.data_caches
            ]
        )
        self.n_data = len(self.data_caches[0])

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # astype -> float32
        data_dict = {
            data_type: value.iloc[idx].values.astype("float32")
            for data_type, value in self.data_caches
        }

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict

    @property
    def index(self):
        return self.data_caches[0].index
