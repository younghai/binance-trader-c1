import torch
import torch.nn as nn
from abc import abstractmethod
from .utils import save_model, load_model, weights_init, CRITERIONS
from datasets.dataset import Dataset
from torch.utils.data import DataLoader
from copy import copy
import os
from itertools import chain
import fire
from .backbone import PredictorV1

CRITERIONS = {"l1": nn.L1Loss, "l2": nn.MSELoss, "ce": nn.CrossEntropyLoss}

DATA_CONFIG = {
    "checkpoint_dir": "./check_points",
    "test_output_dir": "./test_outputs",
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
}


def _mutate_config_path(data_config, exp_dir):
    for key in [
        "checkpoint_dir",
        "test_output_dir",
    ]:
        if data_config[key][0] != "/":
            data_config[key] = os.path.join(exp_dir, data_config[key])

        if "_dir" in key:
            os.makedirs(data_config[key], exist_ok=True)

    return data_config


class AbstractTrainer:
    def __init__(
        self,
        data_dir,
        test_data_dir,
        d_config={},
        m_config={},
        exp_dir="./experiments",
        device="cuda",
    ):
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.exp_dir = exp_dir
        self.device = device

        self.data_config, self.model_config = self._build_config(
            d_config=d_config, m_config=m_config
        )
        self.train_data_loader, self.test_data_loader = self._build_data_loader()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()

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

    def _build_data_loaders(self):
        transforms = {}

        # Define: dataset
        train_dataset = Dataset(
            data_dir=self.data_dir,
            transforms=transforms,
            load_files=self.data_config["load_files"],
        )

        test_dataset = Dataset(
            data_dir=self.test_data_dir,
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

        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.model_config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.model_config["batch_size"] // 2,
        )

        return train_data_loader, test_data_loader

    def _load_model(self, model):
        # load model (inplace)
        load_model(
            model=model,
            dir=self.data_config["checkpoint_dir"],
            strict=self.model_config["load_strict"],
        )

    def _save_model(self, model, epoch):
        save_model(model=model, dir=self.data_config["checkpoint_dir"], epoch=epoch)

    def _build_model(self):
        # Define  model
        model = PredictorV1(
            n_classes=self.model_config["n_classes"],
            n_blocks=self.model_config["n_blocks"],
            n_block_layers=self.model_config["n_block_layers"],
            growth_rate=self.model_config["growth_rate"],
            dropout=self.model_config["dropout"],
            channel_reduction=self.model_config["channel_reduction"],
            activation=self.model_config["activation"],
            normalization=self.model_config["normalization"],
            seblock=self.model_config["seblock"],
            sablock=self.model_config["sablock"],
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

    @abstractmethod
    def step(self, data_dict):
        pass

    @abstractmethod
    def test(self, epoch, iter, losses):
        pass

    @abstractmethod
    def train(self):
        pass
