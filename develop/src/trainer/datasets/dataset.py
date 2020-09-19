from torch.utils.data import Dataset as _Dataset
from typing import Dict, List, Callable
from glob import glob
import os
import numpy as np


FILENAME_TEMPLATE = {
    "X": "{}.npy",
    "Y": "{}.npy",
}


class Dataset(_Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Dict[str, Callable],
        load_files: List[str] = ["X", "Y"],
    ):
        self.filename_template = {
            data_type: FILENAME_TEMPLATE[data_type] for data_type in load_files
        }
        self.dirs = {
            data_type: os.path.join(data_dir, data_type)
            for data_type in self.filename_template.keys()
        }
        self.transforms = transforms
        self.n_data = len(glob(os.path.join(list(self.dirs.values())[0], "*.npy")))

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        data_dict = {
            data_type: np.load(
                os.path.join(
                    self.dirs[data_type], self.filename_template[data_type].format(idx)
                )
            )
            for data_type in self.filename_template.keys()
        }

        # astype -> float32
        for data_type in self.filename_template.keys():
            data_dict[data_type] = data_dict[data_type].astype("float32")

        # transform
        for data_type, transform in self.transforms.items():
            data_dict[data_type] = transform(data_dict[data_type])

        return data_dict
