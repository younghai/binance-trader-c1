from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc


FILENAME_TEMPLATE = {
    "X": "X.parquet.zstd",
    "Y": "Y.parquet.zstd",
}


def build_X_and_BX(features, base_feature_assets, drop_feature_assets):
    BX = features[base_feature_assets]

    trainable_assets = [
        asset
        for asset in features.columns.levels[0]
        if asset not in drop_feature_assets
    ]

    return features[trainable_assets], BX


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        base_feature_assets: List[str],
        drop_feature_assets: List[str],
        asset_to_id: Dict[str, int],
        lookback_window: int = 30,
    ):
        print("[+] Start to build dataset")
        self.data_caches = {}

        # Build inputs
        self.data_caches["X"], self.data_caches["BX"] = build_X_and_BX(
            features=(
                pd.read_parquet(
                    os.path.join(data_dir, FILENAME_TEMPLATE["X"]), engine="pyarrow"
                ).astype("float32")
            ),
            base_feature_assets=base_feature_assets,
            drop_feature_assets=drop_feature_assets,
        )

        assert (self.data_caches["BX"].index == self.data_caches["X"].index).all()

        self.index = []
        for asset in tqdm(self.data_caches["X"].columns.levels[0]):
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
            pd.read_parquet(
                os.path.join(data_dir, FILENAME_TEMPLATE["Y"]), engine="pyarrow",
            )
            .sort_index()
            .stack()
            .reindex(self.index)
        ).astype("float32")

        self.transforms = transforms
        self.n_data = len(self.index)
        self.lookback_window = lookback_window
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
