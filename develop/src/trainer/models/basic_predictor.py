import os
import shutil
import fire
import json
from tqdm import tqdm
import pandas as pd
from copy import copy
from contextlib import contextmanager
from abc import abstractmethod

import torch
import torch.nn as nn
from common_utils_dev import load_text, to_abs_path, get_parent_dir
from .utils import save_model, load_model, weights_init
from .criterions import CRITERIONS
from ..datasets.dataset import Dataset
from torch.utils.data import DataLoader
from trainer.models import backbones

COMMON_CONFIG = {
    "data_dir": to_abs_path(__file__, "../../../storage/dataset/v001/train"),
    "exp_dir": to_abs_path(__file__, "../../../storage/experiments/v001"),
    "test_data_dir": to_abs_path(__file__, "../../../storage/dataset/v001/test"),
}


DATA_CONFIG = {
    "checkpoint_dir": "./check_point",
    "generate_output_dir": "./generated_output",
    "winsorize_threshold": None,
    "base_feature_assets": ["BTC-USDT", "ETH-USDT"],
    "drop_feature_assets": [],
}

MODEL_CONFIG = {
    "lookback_window": 60,
    "batch_size": 1024,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.99,
    "epochs": 5,
    "print_epoch": 1,
    "print_iter": 25,
    "save_epoch": 1,
    "criterion": "fl",
    "criterion_params": {},
    "load_strict": False,
    "model_name": "BackboneV1",
    "model_params": {
        "in_channels": 102,
        "n_class_qay": 10,
        "n_class_qby": 10,
        "n_blocks": 4,
        "n_block_layers": 8,
        "growth_rate": 12,
        "dropout": 0.2,
        "channel_reduction": 0.5,
        "activation": "selu",
        "normalization": "gn",
        "seblock": True,
        "sablock": True,
    },
}


def _mutate_config_path(data_config, exp_dir):
    for key in ["checkpoint_dir", "generate_output_dir"]:
        if data_config[key][0] != "/":
            data_config[key] = os.path.join(exp_dir, data_config[key])

        if "_dir" in key:
            os.makedirs(data_config[key], exist_ok=True)

    return data_config


class BasicPredictor:
    def __init__(
        self,
        data_dir=COMMON_CONFIG["data_dir"],
        test_data_dir=COMMON_CONFIG["test_data_dir"],
        d_config={},
        m_config={},
        exp_dir=COMMON_CONFIG["exp_dir"],
        device="cuda",
        pin_memory=False,
        num_workers=8,
        mode="train",
        default_d_config=DATA_CONFIG,
        default_m_config=MODEL_CONFIG,
    ):
        assert mode in ("train", "test", "predict")
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.exp_dir = exp_dir
        self.device = device
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.mode = mode

        self.data_config, self.model_config = self._build_config(
            d_config=d_config,
            m_config=m_config,
            default_d_config=default_d_config,
            default_m_config=default_m_config,
        )

        self.model = self._build_model()

        self.iterable_train_data_loader = None
        self.iterable_test_data_loader = None

        if mode in ("train", "test"):
            self.asset_to_id = self._build_asset_to_id()

        if mode == "train":
            self.train_data_loader, self.test_data_loader = self._build_data_loaders(
                mode=mode
            )
            self.optimizer = self._build_optimizer()
            self.criterion = self._build_criterion()
        elif mode == "test":
            _, self.test_data_loader = self._build_data_loaders(mode=mode)

        # Store params
        if mode == "train":
            self._store_params()

    def _store_params(self):
        params = {
            "data_dir": self.data_dir,
            "test_data_dir": self.test_data_dir,
            "model_config": self.model_config,
            "data_config": self.data_config,
            "asset_to_id": self.asset_to_id,
        }
        with open(os.path.join(self.exp_dir, f"params.json"), "w") as f:
            json.dump(params, f)

        # Copy files from dataset
        for base_file, target_file in [
            (
                os.path.join(get_parent_dir(self.data_dir), "bins.csv"),
                os.path.join(self.exp_dir, "bins.csv"),
            ),
            (
                os.path.join(get_parent_dir(self.data_dir), "params.json"),
                os.path.join(self.exp_dir, "dataset_params.json"),
            ),
            (
                os.path.join(get_parent_dir(self.data_dir), "scaler.pkl"),
                os.path.join(self.exp_dir, "scaler.pkl"),
            ),
        ]:
            shutil.copy(base_file, target_file)

        print(f"[+] Params are stored")

    def _list_tradable_assets(self, drop_feature_assets):
        tradable_assets = load_text(
            os.path.join(
                self.data_dir.replace("/test", "").replace("/train", ""),
                "tradable_coins.txt",
            )
        )
        tradable_assets = [
            tradable_asset
            for tradable_asset in tradable_assets
            if tradable_asset not in drop_feature_assets
        ]

        return tradable_assets

    def _build_config(self, d_config, m_config, default_d_config, default_m_config):
        # refine path with exp_dirs
        data_config = copy(default_d_config)
        model_config = copy(default_m_config)
        if not set(m_config.keys()).issubset(set(model_config.keys())):
            raise ValueError(f"{set(m_config.keys()) - set(model_config.keys())}")

        if not set(d_config.keys()).issubset(set(data_config.keys())):
            raise ValueError(f"{set(d_config.keys()) - set(data_config.keys())}")

        data_config = {**data_config, **d_config}

        model_params = {
            **model_config.pop("model_params", {}),
            **m_config.pop("model_params", {}),
        }
        model_config = {**model_config, **m_config, **{"model_params": model_params}}

        data_config = _mutate_config_path(data_config=data_config, exp_dir=self.exp_dir)

        # Mutate model_params' n_assets
        if "n_assets" not in model_config["model_params"]:
            n_assets = len(
                self._list_tradable_assets(
                    drop_feature_assets=data_config["drop_feature_assets"]
                )
            )
            model_config["model_params"]["n_assets"] = n_assets

        return data_config, model_config

    def _build_asset_to_id(self):
        tradable_assets = self._list_tradable_assets(
            drop_feature_assets=self.data_config["drop_feature_assets"]
        )
        asset_to_id = {
            tradable_asset: idx for idx, tradable_asset in enumerate(tradable_assets)
        }
        return asset_to_id

    def _build_transfroms(self):
        return {}

    def _build_data_loaders(self, mode):
        assert mode in ("train", "test")
        transforms = self._build_transfroms()

        # Build base params
        base_dataset_params = {
            "transforms": transforms,
            "lookback_window": self.model_config["lookback_window"],
            "winsorize_threshold": self.data_config["winsorize_threshold"],
            "base_feature_assets": self.data_config["base_feature_assets"],
            "drop_feature_assets": self.data_config["drop_feature_assets"],
            "asset_to_id": self.asset_to_id,
        }

        base_data_loader_params = {
            "batch_size": self.model_config["batch_size"],
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
        }

        # Build dataset & data_loader
        test_dataset = Dataset(data_dir=self.test_data_dir, **base_dataset_params)

        train_data_loader = None
        if mode == "train":
            # Define: dataset
            train_dataset = Dataset(data_dir=self.data_dir, **base_dataset_params)

            # Define data_loader
            train_data_loader = DataLoader(
                dataset=train_dataset, shuffle=True, **base_data_loader_params
            )

            test_data_loader = DataLoader(
                dataset=test_dataset, shuffle=True, **base_data_loader_params
            )

        if mode == "test":
            test_data_loader = DataLoader(
                dataset=test_dataset, shuffle=False, **base_data_loader_params
            )

        return train_data_loader, test_data_loader

    def _load_model(self, model):
        # load model (inplace)
        self.last_epoch = load_model(
            model=model,
            dir=self.data_config["checkpoint_dir"],
            strict=self.model_config["load_strict"],
            device=self.device,
        )
        if self.mode in ("test", "predict"):
            assert self.last_epoch != -1

    def _save_model(self, model, epoch):
        save_model(model=model, dir=self.data_config["checkpoint_dir"], epoch=epoch)

    def _build_model(self):
        # Define  model
        model = getattr(backbones, self.model_config["model_name"])(
            **self.model_config["model_params"]
        )

        # Init model's weights
        model.apply(weights_init)

        # Setup device
        if torch.cuda.device_count() > 1:
            print("Notice: use ", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)
        else:
            print(f"Notice: use {self.device}")

        # Load models
        self._load_model(model=model)
        model.to(self.device)

        return model

    def _build_optimizer(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.model_config["lr"],
            (self.model_config["beta1"], self.model_config["beta2"]),
        )

        return optimizer

    def _build_criterion(self):
        return CRITERIONS[self.model_config["criterion"]](
            **self.model_config["criterion_params"]
        ).to(self.device)

    def _generate_train_data_dict(self):
        if self.iterable_train_data_loader is None:
            self.iterable_train_data_loader = iter(self.train_data_loader)

        # Pick data
        try:
            train_data_dict = next(self.iterable_train_data_loader)
        except StopIteration:
            self.iterable_train_data_loader = iter(self.train_data_loader)
            train_data_dict = next(self.iterable_train_data_loader)

        train_data_dict = {
            key: value.to(self.device) for key, value in train_data_dict.items()
        }
        return train_data_dict

    def _generate_test_data_dict(self):
        if self.iterable_test_data_loader is None:
            self.iterable_test_data_loader = iter(self.test_data_loader)

        # Pick data
        try:
            test_data_dict = next(self.iterable_test_data_loader)
        except StopIteration:
            self.iterable_test_data_loader = iter(self.test_data_loader)
            test_data_dict = next(self.iterable_test_data_loader)

        test_data_dict = {
            key: value.to(self.device) for key, value in test_data_dict.items()
        }
        return test_data_dict

    @abstractmethod
    def _step(self, train_data_dict):
        pass

    @abstractmethod
    def train(self):
        """
        Train model
        """
        pass

    @abstractmethod
    def generate(self, save_dir=None):
        """
        Generate historical predictions csv
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict
        """
