import os
import fire
import json
from tqdm import tqdm
import pandas as pd
from copy import copy
from contextlib import contextmanager
from abc import abstractmethod

import torch
import torch.nn as nn
from .utils import save_model, load_model, weights_init
from .criterions import CRITERIONS
from ..datasets.dataset import Dataset
from torch.utils.data import DataLoader
from trainer.models import backbones


DATA_CONFIG = {
    "checkpoint_dir": "./check_point",
    "generate_output_dir": "./generated_output",
    "winsorize_threshold": None,
}

MODEL_CONFIG = {
    "lookback_window": 60,
    "batch_size": 1024,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.99,
    "epochs": 100,
    "print_epoch": 1,
    "print_iter": 25,
    "save_epoch": 1,
    "criterion": "fl",
    "load_strict": False,
    "model_name": "BackboneV1",
    "model_params": {
        "in_channels": 320,
        "n_assets": 32,
        "n_class_per_asset": 4,
        "n_blocks": 3,
        "n_block_layers": 16,
        "growth_rate": 12,
        "dropout": 0.2,
        "channel_reduction": 0.5,
        "activation": "relu",
        "normalization": "bn",
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
        data_dir,
        test_data_dir,
        d_config={},
        m_config={},
        exp_dir="./experiments",
        device="cuda",
        pin_memory=True,
        num_workers=16,
        mode="train",
    ):
        assert mode in ("train", "test")
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.exp_dir = exp_dir
        self.device = device
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.mode = mode

        self.data_config, self.model_config = self._build_config(
            d_config=d_config, m_config=m_config
        )
        self.model = self._build_model()
        self.iterable_train_data_loader = None
        self.iterable_test_data_loader = None

        if mode == "train":
            self.train_data_loader, self.test_data_loader = self._build_data_loaders(
                mode=mode
            )
            self.optimizer = self._build_optimizer()
            self.criterion = self._build_criterion()
        elif mode == "test":
            _, self.test_data_loader = self._build_data_loaders(mode=mode)

        # Store params
        self._store_params()

    def _store_params(self):
        params = {
            "data_dir": self.data_dir,
            "test_data_dir": self.test_data_dir,
            "model_config": self.model_config,
        }
        with open(os.path.join(self.exp_dir, f"params.csv"), "w") as f:
            json.dump(params, f)

        print(f"[+] Params are stored")

    def _build_config(self, d_config, m_config):
        # refine path with exp_dirs
        data_config = copy(DATA_CONFIG)
        model_config = copy(MODEL_CONFIG)
        if not set(m_config.keys()).issubset(set(model_config.keys())):
            raise ValueError(f"{set(m_config.keys()) - set(model_config.keys())}")

        if not set(d_config.keys()).issubset(set(data_config.keys())):
            raise ValueError(f"{set(d_config.keys()) - set(data_config.keys())}")

        data_config = {**data_config, **d_config}

        model_params = {
            **model_config.pop("model_params"),
            **m_config.pop("model_params"),
        }
        model_config = {**model_config, **m_config, **{"model_params": model_params}}

        data_config = _mutate_config_path(data_config=data_config, exp_dir=self.exp_dir)

        return data_config, model_config

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
        )

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
        return CRITERIONS[self.model_config["criterion"]]().to(self.device)

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

    def generate(self, save_dir=None):
        assert self.mode in ("test")
        self.model.eval()

        if save_dir is None:
            save_dir = self.data_config["generate_output_dir"]

        index = self.test_data_loader.dataset.index

        predictions = []
        labels = []
        for _ in tqdm(range(len(self.test_data_loader))):
            test_data_dict = self._generate_test_data_dict()

            X, Y = test_data_dict["X"], test_data_dict["Y"]

            y_preds = self.model(X)
            B, _, _ = y_preds.size()

            predictions += y_preds.argmax(dim=-1).view(B, -1).cpu().tolist()
            labels += Y.view(B, -1).cpu().tolist()

        pd.DataFrame(predictions, index=index).to_csv(
            os.path.join(save_dir, "predictions.csv")
        )
        pd.DataFrame(labels, index=index).to_csv(os.path.join(save_dir, "labels.csv"))

    def predict(self, X):
        self.model.eval()
        return self.model(X.to(self.device)).argmax(dim=-1).cpu()
