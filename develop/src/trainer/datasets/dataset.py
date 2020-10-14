from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable, Optional
import os
import numpy as np
import pandas as pd


FILENAME_TEMPLATE = {"X": "X.csv", "QAY": "QAY.csv", "QBY": "QBY.csv"}
CONFIG = {
    "base_feature_assets": ["BTC-USDT", "ETH-BTC"],
}


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        lookback_window: int = 60,
        winsorize_threshold: Optional[int] = None,
        base_feature_assets: List[str] = CONFIG["base_feature_assets"],
    ):
        self.data_caches = {}
        x = pd.read_csv(
            os.path.join(data_dir, FILENAME_TEMPLATE[data_type]),
            header=[0, 1],
            index_col=0,
            compression="gzip",
        )

        self.data_caches["BX"] = x[base_feature_assets]
        self.data_caches["X"] = x[feature_assets].stack(level=0)
        self.data_caches = {
            data_type: pd.read_csv(
                os.path.join(data_dir, FILENAME_TEMPLATE[data_type]),
                header=0,
                index_col=0,
                compression="gzip",
            ).stack()
            for data_type in ["QAY", "QBY"]
        }

        # Check if all index are all same.
        assert (self.data_caches["X"].index == self.data_caches["QAY"].index).all()
        assert (self.data_caches["X"].index == self.data_caches["QBY"].index).all()

        # Mask data without None
        mask_index = (
            self.data_caches["X"].dropna().index
            & self.data_caches["QAY"].dropna().index
            & self.data_caches["QBY"].dropna().index
        ).sort_index()

        self.data_caches["BX"] = self.data_caches["BX"].reindex(mask_index.levels[0])
        self.data_caches["X"] = self.data_caches["X"].reindex(mask_index)
        self.data_caches["QAY"] = self.data_caches["QAY"].reindex(mask_index)
        self.data_caches["QBY"] = self.data_caches["QBY"].reindex(mask_index)

        self.transforms = transforms

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

        data_dict["QY"] = (
            self.data_caches["QY"]
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
