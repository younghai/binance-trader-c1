from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable, Optional
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from common_utils import load_text


FILENAME_TEMPLATE = {
    "X": "X.parquet.zstd",
    "QAY": "QAY.parquet.zstd",
    "QBY": "QBY.parquet.zstd",
}
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
        print("[+] Start to build dataset")
        self.data_caches = {}
        self.data_caches["X"] = pd.read_parquet(
            os.path.join(data_dir, FILENAME_TEMPLATE["X"])
        )
        self.data_caches["BX"] = self.data_caches["X"][base_feature_assets]

        self.index = []
        for asset in tqdm(self.data_caches["X"].columns.levels[0]):
            self.index += (
                self.data_caches["X"][[asset]]
                .dropna()
                .iloc[lookback_window - 1 :]
                .sort_index()
                .stack(level=0)
                .index.to_list()
            )
        self.index = pd.Index(self.index)

        for data_type in ["QAY", "QBY"]:
            self.data_caches[data_type] = (
                pd.read_parquet(os.path.join(data_dir, FILENAME_TEMPLATE[data_type]),)
                .sort_index()
                .stack()
                .reindex(self.index)
                .astype(int)
            )

        self.transforms = transforms
        self.n_data = len(self.index)
        self.lookback_window = lookback_window
        self.winsorize_threshold = winsorize_threshold

        tradable_assets = load_text(
            os.path.join(
                data_dir.split("/test")[0].split("/train")[0], "tradable_coins.txt"
            )
        )
        self.asset_to_id = {
            tradable_asset: idx for idx, tradable_asset in enumerate(tradable_assets)
        }

        del tradable_assets
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
                self.data_caches["BX"].iloc[boundary_index - 59 : boundary_index + 1],
                self.data_caches["X"][self.index[idx][1]].iloc[
                    boundary_index - 59 : boundary_index + 1
                ],
            ],
            axis=1,
        ).astype("float32")

        data_dict["X"] = np.swapaxes(concat_df.values, 0, 1,)

        if self.winsorize_threshold is not None:
            data_dict["X"] = data_dict["X"].clip(
                -self.winsorize_threshold, self.winsorize_threshold
            )

        data_dict["QAY"] = self.data_caches["QAY"].iloc[idx]

        data_dict["QBY"] = self.data_caches["QBY"].iloc[idx]

        data_dict["ID"] = self.asset_to_id[self.index[idx][1]]

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        del concat_df

        return data_dict
