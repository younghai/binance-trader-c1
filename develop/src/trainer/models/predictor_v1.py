import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from .basic_predictor import BasicPredictor
from common_utils import to_parquet, to_abs_path

COMMON_CONFIG = {
    "data_dir": to_abs_path(__file__, "../../../storage/dataset/dataset_60m_v1/train"),
    "exp_dir": to_abs_path(__file__, "../../../storage/experiments/v001"),
    "test_data_dir": to_abs_path(
        __file__, "../../../storage/dataset/dataset_60m_v1/test"
    ),
}

DATA_CONFIG = {
    "checkpoint_dir": "./check_point",
    "generate_output_dir": "./generated_output",
    "winsorize_threshold": None,
    "base_feature_assets": ["BTC-USDT", "ETH-BTC"],
    "drop_feature_assets": [],
}

MODEL_CONFIG = {
    "lookback_window": 60,
    "batch_size": 1024,
    "lr": 0.001,
    "beta1": 0.5,
    "beta2": 0.99,
    "epochs": 10,
    "print_epoch": 1,
    "print_iter": 25,
    "save_epoch": 1,
    "criterion": "fl",
    "criterion_params": {},
    "load_strict": False,
    "model_name": "BackboneV1",
    "model_params": {
        "in_channels": 111,
        "n_class_qay": 10,
        "n_class_qby": 10,
        "n_blocks": 4,
        "n_block_layers": 8,
        "growth_rate": 12,
        "dropout": 0.2,
        "channel_reduction": 0.5,
        "activation": "tanhexp",
        "normalization": "gn",
        "seblock": True,
        "sablock": True,
    },
}


class PredictorV1(BasicPredictor):
    """
    Functions:
        train(): train the model with train_data
        generate(save_dir: str): generate predictions & labels with test_data
        predict(X: torch.Tensor): gemerate prediction with given data
    """

    def __init__(
        self,
        data_dir=COMMON_CONFIG["data_dir"],
        test_data_dir=COMMON_CONFIG["test_data_dir"],
        d_config={},
        m_config={},
        exp_dir=COMMON_CONFIG["exp_dir"],
        device="cuda",
        pin_memory=True,
        num_workers=8,
        mode="train",
        default_d_config=DATA_CONFIG,
        default_m_config=MODEL_CONFIG,
    ):
        super().__init__(
            data_dir=data_dir,
            test_data_dir=test_data_dir,
            d_config=d_config,
            m_config=m_config,
            exp_dir=exp_dir,
            device=device,
            pin_memory=pin_memory,
            num_workers=num_workers,
            mode=mode,
            default_d_config=default_d_config,
            default_m_config=default_m_config,
        )

    def _compute_train_loss(self, train_data_dict):
        # Set train mode
        self.model.train()
        self.model.zero_grad()

        # Set loss
        preds_qay, preds_qby = self.model(
            x=train_data_dict["X"], id=train_data_dict["ID"]
        )

        # Y loss
        loss = self.criterion(preds_qay, train_data_dict["QAY"])
        loss += self.criterion(preds_qby, train_data_dict["QBY"])

        return loss

    def _compute_test_loss(self, test_data_dict):
        # Set eval mode
        self.model.eval()

        # Set loss
        preds_qay, preds_qby = self.model(
            x=test_data_dict["X"], id=test_data_dict["ID"]
        )

        # Y loss
        loss = self.criterion(preds_qay, test_data_dict["QAY"])
        loss += self.criterion(preds_qby, test_data_dict["QBY"])
        loss = loss.detach()

        return loss

    def _step(self):
        train_data_dict = self._generate_train_data_dict()
        loss = self._compute_train_loss(train_data_dict=train_data_dict)
        loss.backward()
        self.optimizer.step()

        return loss

    def _display_info(self, train_loss):
        # Print loss info
        test_data_dict = self._generate_test_data_dict()
        test_loss = self._compute_test_loss(test_data_dict)

        print(f""" [+] train_loss: {train_loss:.2f} | test_loss: {test_loss:.2f} """)

    def train(self):
        for epoch in range(self.model_config["epochs"]):
            if epoch <= self.last_epoch:
                continue

            for iter_ in tqdm(range(len(self.train_data_loader))):
                # Optimize
                train_loss = self._step()

                # Display losses
                if epoch % self.model_config["print_epoch"] == 0:
                    if iter_ % self.model_config["print_iter"] == 0:
                        self._display_info(train_loss=train_loss)

            # Store the check-point
            if (epoch % self.model_config["save_epoch"] == 0) or (
                epoch == self.model_config["epochs"] - 1
            ):
                self._save_model(model=self.model, epoch=epoch)

    def generate(self, save_dir=None, test=False):
        assert self.mode in ("test")
        self.model.eval()

        if save_dir is None:
            save_dir = self.data_config["generate_output_dir"]

        index = self.test_data_loader.dataset.index

        qay_predictions = []
        qay_probabilities = []
        qay_labels = []

        qby_predictions = []
        qby_probabilities = []
        qby_labels = []
        for idx in tqdm(range(len(self.test_data_loader))):
            test_data_dict = self._generate_test_data_dict()

            preds_qay, preds_qby = self.model(
                x=test_data_dict["X"], id=test_data_dict["ID"]
            )

            qay_predictions += preds_qay.argmax(dim=-1).view(-1).cpu().tolist()
            qay_probabilities += preds_qay.max(dim=-1).values.view(-1).cpu().tolist()
            qay_labels += test_data_dict["QAY"].view(-1).cpu().tolist()

            qby_predictions += preds_qby.argmax(dim=-1).view(-1).cpu().tolist()
            qby_probabilities += preds_qby.max(dim=-1).values.view(-1).cpu().tolist()
            qby_labels += test_data_dict["QBY"].view(-1).cpu().tolist()

            if test is True:
                index = index[: 100 * self.model_config["batch_size"]]
                if idx == 99:
                    break

        # Store signals
        for data_type, data in [
            ("qay_predictions", qay_predictions),
            ("qay_probabilities", qay_probabilities),
            ("qay_labels", qay_labels),
            ("qby_predictions", qby_predictions),
            ("qby_probabilities", qby_probabilities),
            ("qby_labels", qby_labels),
        ]:
            to_parquet(
                df=pd.Series(data, index=index).sort_index().unstack(),
                path=os.path.join(save_dir, f"{data_type}.parquet.zstd"),
            )

    def predict(self, X, id):
        self.model.eval()
        preds_qay, preds_qby = self.model(
            x=X.to(self.device), id=torch.tensor(id).to(self.device)
        )

        return {
            "qay_prediction": preds_qay.argmax(dim=-1).cpu(),
            "qay_probability": preds_qay.max(dim=-1).cpu(),
            "qby_prediction": preds_qby.argmax(dim=-1).cpu(),
            "qby_probability": preds_qby.max(dim=-1).cpu(),
        }


if __name__ == "__main__":
    import fire

    fire.Fire(PredictorV1)
