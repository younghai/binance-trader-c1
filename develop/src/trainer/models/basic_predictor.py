import fire
import os
import pandas as pd
from copy import copy
from contextlib import contextmanager
from abc import abstractmethod

import torch
import torch.nn as nn
from .utils import save_model, load_model, weights_init, CRITERIONS
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
from trainer.models import backbones

CRITERIONS = {"l1": nn.L1Loss, "l2": nn.MSELoss, "ce": nn.CrossEntropyLoss}

DATA_CONFIG = {
    "checkpoint_dir": "./check_point",
    "generate_output_dir": "./generate_output",
    "load_files": ["X", "Y"],
}

MODEL_CONFIG = {
    "batch_size": 64,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.99,
    "epochs": 100,
    "print_epoch": 1,
    "test_epoch": 1,
    "save_epoch": 1,
    "print_iter": 10,
    "test_iter": 500,
    "criterion": "ce",
    "model_name": "BackboneV1",
    "model_params": {
        "n_class_per_asset": 4,
        "n_classes": 120,
        "n_blocks": 3,
        "n_block_layers": 6,
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
        mode="train",
    ):
        assert mode in ("train", "test")
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.exp_dir = exp_dir
        self.device = device
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

    def _build_config(self, d_config, m_config):
        # refine path with exp_dirs
        data_config = copy(DATA_CONFIG)
        model_config = copy(MODEL_CONFIG)
        if not set(m_config.keys()).issubset(set(model_config.keys())):
            raise ValueError(f"{set(m_config.keys()) - set(model_config.keys())}")

        if not set(d_config.keys()).issubset(set(data_config.keys())):
            raise ValueError(f"{set(d_config.keys()) - set(data_config.keys())}")

        data_config = {**data_config, **d_config}
        model_config = {**model_config, **m_config}

        data_config = _mutate_config_path(data_config=data_config, exp_dir=self.exp_dir)

        return data_config, model_config

    def _build_transfroms(self):
        return {}

    def _build_data_loaders(self, mode):
        assert mode in ("train", "test")
        transforms = self._build_transfroms()

        test_dataset = Dataset(
            data_dir=self.test_data_dir,
            transforms=transforms,
            load_files=self.data_config["load_files"],
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.model_config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.model_config["batch_size"] // 2,
        )

        train_data_loader = None
        if mode == "train":
            # Define: dataset
            train_dataset = Dataset(
                data_dir=self.data_dir,
                transforms=transforms,
                load_files=self.data_config["load_files"],
            )

            # Define data_loader
            train_data_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.model_config["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=self.model_config["batch_size"] // 2,
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

    @contextmanager
    def _iterable_data_loader_with_false_shuffle(self, *args, **kwds):
        # Backup variables
        backup_shuffle = self.test_data_loader.shuffle

        self.test_data_loader.shuffle = False
        self.iterable_train_data_loader = iter(self.train_data_loader)
        try:
            yield self.iterable_train_data_loader
        finally:
            # Set to original variable
            self.test_data_loader.shuffle = backup_shuffle
            self.iterable_test_data_loader = None

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
        if save_dir is None:
            save_dir = self.data_config["generate_output_dir"]

        index = self.test_data_loader.dataset.index

        # Mutate shuffle
        with self._iterable_data_loader_with_false_shuffle() as _:

            predictions = []
            labels = []
            for _ in len(self.test_data_loader):
                test_data_dict = self._generate_test_data_dict()

                X, Y = test_data_dict["X"], test_data_dict["Y"]

                y_preds = self.model(X)
                predictions += y_preds.argmax(dim=-1).view(-1).cpu().tolist()
                labels += Y.view(-1).cpu().tolist()

            pd.concat(
                [
                    pd.Series(predictions, index=index).rename("prediction"),
                    pd.Series(labels, index=index).rename("label"),
                ]
            ).to_csv(os.path.join(save_dir, "predictions.csv"))

    def predict(self, X):
        return self.model(X.to(self.device)).argmax(dim=-1).cpu()
