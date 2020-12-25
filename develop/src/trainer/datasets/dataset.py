from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable, Optional
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from common_utils_dev import load_text
import gc


FILENAME_TEMPLATE = {
    "X": "X.parquet.zstd",
    "Y": "Y.parquet.zstd",
}


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        base_feature_assets: List[str],
        drop_feature_assets: List[str],
        asset_to_id: Dict[str, int],
        lookback_window: int = 60,
        winsorize_threshold: int = 6,
    ):
        print("[+] Start to build dataset")
        self.data_caches = {}

        # Build inputs
        assert winsorize_threshold is not None
        self.data_caches["X"] = (
            pd.read_parquet(
                os.path.join(data_dir, FILENAME_TEMPLATE["X"]), engine="pyarrow"
            )
            .astype("float32")
            .clip(-winsorize_threshold, winsorize_threshold)
        ) / winsorize_threshold

        self.data_caches["BX"] = self.data_caches["X"][base_feature_assets]

        trainable_assets = [
            asset
            for asset in self.data_caches["X"].columns.levels[0]
            if asset not in drop_feature_assets
        ]
        self.data_caches["X"] = self.data_caches["X"][trainable_assets]

        self.index = []
        for asset in tqdm(trainable_assets):
            self.index += [
                (index, asset)
                for index in self.data_caches["X"][[asset]]
                .dropna()
                .iloc[lookback_window - 1 :]
                .index
            ]

        self.index = pd.Index(self.index)
        gc.collect()

        # Build labels
        self.data_caches["Y"] = (
            (
                pd.read_parquet(
                    os.path.join(data_dir, FILENAME_TEMPLATE["Y"]), engine="pyarrow",
                )
                .sort_index()
                .stack()
                .reindex(self.index)
            )
            .astype("float32")
            .clip(-winsorize_threshold, winsorize_threshold)
            / winsorize_threshold
        )

        self.transforms = transforms
        self.n_data = len(self.index)
        self.lookback_window = lookback_window
        self.winsorize_threshold = winsorize_threshold
        self.asset_to_id = asset_to_id

        gc.collect()
        print("[+] built dataset")

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        # astype -> Y: int, else: float32
        data_dict = {}

        boundary_index = self.data_caches["X"][self.index[idx][1]].index.get_loc(
            self.index[idx][0]
        )

        # Concat with BX
        concat_df = pd.concat(
            [
                self.data_caches["BX"].iloc[
                    boundary_index - (self.lookback_window - 1) : boundary_index + 1
                ],
                self.data_caches["X"][self.index[idx][1]].iloc[
                    boundary_index - (self.lookback_window - 1) : boundary_index + 1
                ],
            ],
            axis=1,
        )

        data_dict["X"] = np.swapaxes(concat_df.values, 0, 1)

        data_dict["Y"] = self.data_caches["Y"].iloc[idx]

        data_dict["ID"] = self.asset_to_id[self.index[idx][1]]

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        del concat_df

        return data_dict
