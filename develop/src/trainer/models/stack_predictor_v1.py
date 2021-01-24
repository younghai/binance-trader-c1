import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List, Dict
from tqdm import tqdm
from .basic_predictor import BasicPredictor
from .utils import inverse_preprocess_data
from common_utils_dev import to_parquet, to_abs_path, load_json
from trainer.models.predictor_v1 import PredictorV1

COMMON_CONFIG = {
    "data_dir": to_abs_path(__file__, "../../../storage/dataset/dataset/v001/train"),
    "exp_dir": to_abs_path(__file__, "../../../storage/experiments/v001_stack"),
    "test_data_dir": to_abs_path(
        __file__, "../../../storage/dataset/dataset/v001/test"
    ),
    "feature_extractor_dir": to_abs_path(__file__, "../../../storage/experiments/v001"),
}

DATA_CONFIG = {
    "checkpoint_dir": "./check_point",
    "generate_output_dir": "./generated_output",
    "base_feature_assets": ["BTC-USDT"],
}

MODEL_CONFIG = {
    "local_lookback_window": 30,
    "lookback_window": 149,
    "batch_size": 512,
    "lr": 0.0001,
    "epochs": 15,
    "print_epoch": 1,
    "print_iter": 50,
    "save_epoch": 1,
    "criterion": "l2",
    "criterion_params": {},
    "load_strict": False,
    "model_name": "StackBackboneV1",
    "model_params": {
        "in_channels": 86,
        "n_blocks": 4,
        "n_block_layers": 8,
        "growth_rate": 12,
        "dropout": 0.1,
        "channel_reduction": 0.5,
        "activation": "selu",
        "normalization": None,
        "seblock": True,
        "sablock": True,
    },
}


class StackPredictorV1(BasicPredictor):
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
        feature_extractor_dir=COMMON_CONFIG["exp_dir"],
        exp_dir=COMMON_CONFIG["exp_dir"],
        device="cuda",
        pin_memory=False,
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

        self._build_feature_extractor_params(
            feature_extractor_dir=feature_extractor_dir
        )
        self._build_feature_extractor(feature_extractor_dir=feature_extractor_dir)
        assert (
            self.feature_extractor_params["model_config"]["lookback_window"]
            + self.model_config["local_lookback_window"]
            - 1
            == self.model_config["lookback_window"]
        )

    def _build_feature_extractor_params(self, feature_extractor_dir):
        self.feature_extractor_params = load_json(
            os.path.join(feature_extractor_dir, "trainer_params.json")
        )

    def _build_feature_extractor(self, feature_extractor_dir):
        self.feature_extractor = PredictorV1(
            exp_dir=feature_extractor_dir,
            m_config=self.feature_extractor_params["model_config"],
            d_config=self.feature_extractor_params["data_config"],
            device=self.device,
            mode="predict",
        )

    def _invert_to_sign(self, pred_sign_factor, boundary=0.5):
        return ((pred_sign_factor >= boundary) * 1.0) + (
            (pred_sign_factor < boundary) * -1.0
        )

    def _invert_to_prediction(self, pred_abs_factor, pred_sign_factor):
        multiply = self._invert_to_sign(pred_sign_factor=pred_sign_factor)
        return pred_abs_factor * multiply

    def _extract_features(self, data_dict):
        with torch.no_grad():
            abs_factor_features, sign_factor_features = self.feature_extractor.model(
                x=torch.cat(
                    [
                        data_dict["X"][
                            :,
                            :,
                            idx : self.feature_extractor_params["model_config"][
                                "lookback_window"
                            ]
                            + idx,
                        ]
                        for idx in range(self.model_config["local_lookback_window"])
                    ],
                    dim=0,
                ),
                id=torch.cat(
                    [
                        data_dict["ID"]
                        for _ in range(self.model_config["local_lookback_window"])
                    ],
                    dim=0,
                ),
            )

            abs_factor_features = torch.stack(
                abs_factor_features.squeeze().split(data_dict["ID"].size()[0], dim=0),
                dim=-1,
            )
            sign_factor_features = torch.stack(
                sign_factor_features.squeeze().split(data_dict["ID"].size()[0], dim=0),
                dim=-1,
            )
            return (
                abs_factor_features,
                sign_factor_features,
            )

    def _build_stacked_features(self, data_dict):
        abs_factor_features, sign_factor_features = self._extract_features(
            data_dict=data_dict
        )

        return torch.cat(
            [
                data_dict["X"][:, :, -self.model_config["local_lookback_window"] :],
                abs_factor_features.unsqueeze(dim=1),
                sign_factor_features.unsqueeze(dim=1),
            ],
            dim=1,
        )

    def _compute_train_loss(self, train_data_dict):
        # Set train mode
        self.model.train()
        self.model.zero_grad()

        # Set loss
        x = self._build_stacked_features(data_dict=train_data_dict)

        (
            pred_abs_factor,
            pred_sign_factor,
            pred_abs_error_factor,
            pred_sign_error_factor,
        ) = self.model(x=x, id=train_data_dict["ID"],)

        # Y loss
        loss = self.criterion(pred_abs_factor, train_data_dict["Y"].view(-1).abs()) * 10
        loss += self.binary_cross_entropy(
            pred_sign_factor, (train_data_dict["Y"].view(-1) >= 0) * 1.0
        )

        loss += (
            self.criterion(
                pred_abs_error_factor,
                (
                    train_data_dict["Y"].view(-1).abs() - x[:, -2, -1].view(-1).abs()
                ).abs(),
            )
            * 2
        )
        loss += (
            self.binary_cross_entropy(
                pred_sign_error_factor,
                (
                    (
                        self._invert_to_sign(
                            train_data_dict["Y"].view(-1), boundary=0.0
                        )
                        * self._invert_to_sign(x[:, -1, -1].view(-1), boundary=0.5)
                    )
                    >= 0
                )
                * 1.0,
            )
            * 0.2
        )

        return (
            loss,
            self._invert_to_prediction(
                pred_abs_factor=pred_abs_factor, pred_sign_factor=pred_sign_factor
            ),
        )

    def _compute_test_loss(self, test_data_dict):
        # Set eval mode
        self.model.eval()

        # Set loss
        x = self._build_stacked_features(data_dict=test_data_dict)

        (
            pred_abs_factor,
            pred_sign_factor,
            pred_abs_error_factor,
            pred_sign_error_factor,
        ) = self.model(x=x, id=test_data_dict["ID"])

        # Y loss
        loss = self.criterion(pred_abs_factor, test_data_dict["Y"].view(-1).abs()) * 10
        loss += self.binary_cross_entropy(
            pred_sign_factor, (test_data_dict["Y"].view(-1) >= 0) * 1.0
        )

        loss += (
            self.criterion(
                pred_abs_error_factor,
                (
                    test_data_dict["Y"].view(-1).abs() - x[:, -2, -1].view(-1).abs()
                ).abs(),
            )
            * 2
        )
        loss += (
            self.binary_cross_entropy(
                pred_sign_error_factor,
                (
                    (
                        self._invert_to_sign(test_data_dict["Y"].view(-1), boundary=0.0)
                        * self._invert_to_sign(x[:, -1, -1].view(-1), boundary=0.5)
                    )
                    >= 0
                )
                * 1.0,
            )
            * 0.2
        )

        return (
            loss,
            self._invert_to_prediction(
                pred_abs_factor=pred_abs_factor, pred_sign_factor=pred_sign_factor
            ),
        )

    def _step(self, train_data_dict):
        loss, _ = self._compute_train_loss(train_data_dict=train_data_dict)
        loss.backward()
        self.optimizer.step()

        return loss

    def _display_info(self, train_loss, test_loss, test_predictions, test_labels):
        pred_norm = test_predictions[test_predictions >= 0].abs().mean()
        label_norm = test_labels[test_labels >= 0].abs().mean()

        # Print loss info
        print(
            f""" [+] train_loss: {train_loss:.2f}, test_loss: {test_loss:.2f} | [+] pred_norm: {pred_norm:.2f}, label_norm: {label_norm:.2f}"""
        )

    def _build_abs_bins(self, df):
        abs_bins = {}
        for column in df.columns:
            _, abs_bins[column] = pd.qcut(
                df[column].abs(), 10, labels=False, retbins=True
            )
            abs_bins[column] = np.concatenate([[0], abs_bins[column][1:-1], [np.inf]])

        return pd.DataFrame(abs_bins)

    def _build_probabilities(self, pred_sign_factor):
        return ((pred_sign_factor - 0.5) * 2).abs()

    def train(self):
        for epoch in range(self.model_config["epochs"]):
            if epoch <= self.last_epoch:
                continue

            for iter_ in tqdm(range(len(self.train_data_loader))):
                # Optimize
                train_data_dict = self._generate_train_data_dict()
                train_loss = self._step(train_data_dict=train_data_dict)

                # Display losses
                if epoch % self.model_config["print_epoch"] == 0:
                    if iter_ % self.model_config["print_iter"] == 0:
                        test_data_dict = self._generate_test_data_dict()
                        test_loss, test_predictions = self._compute_test_loss(
                            test_data_dict=test_data_dict
                        )
                        self._display_info(
                            train_loss=train_loss,
                            test_loss=test_loss,
                            test_predictions=test_predictions,
                            test_labels=test_data_dict["Y"],
                        )

            # Store the check-point
            if (epoch % self.model_config["save_epoch"] == 0) or (
                epoch == self.model_config["epochs"] - 1
            ):
                self._save_model(model=self.model, epoch=epoch)

    def generate(self, save_dir=None):
        assert self.mode in ("test")
        self.model.eval()

        if save_dir is None:
            save_dir = self.data_config["generate_output_dir"]

        # Mutate 1 min to handle logic, entry: open, exit: open
        index = self.test_data_loader.dataset.index
        index = index.set_levels(index.levels[0] + pd.Timedelta(minutes=1), level=0)

        predictions = []
        labels = []
        probabilities = []
        for idx in tqdm(range(len(self.test_data_loader))):
            test_data_dict = self._generate_test_data_dict()

            x = self._build_stacked_features(data_dict=test_data_dict)

            (pred_abs_factor, pred_sign_factor, _, _,) = self.model(
                x=x, id=test_data_dict["ID"]
            )

            preds = self._invert_to_prediction(
                pred_abs_factor=pred_abs_factor, pred_sign_factor=pred_sign_factor
            )

            predictions += preds.view(-1).cpu().tolist()
            labels += test_data_dict["Y"].view(-1).cpu().tolist()
            probabilities += (
                self._build_probabilities(pred_sign_factor=pred_sign_factor)
                .view(-1)
                .cpu()
                .tolist()
            )

        predictions = (
            pd.Series(predictions, index=index)
            .sort_index()
            .unstack()[self.dataset_params["labels_columns"]]
        )
        labels = (
            pd.Series(labels, index=index)
            .sort_index()
            .unstack()[self.dataset_params["labels_columns"]]
        )
        probabilities = (
            pd.Series(probabilities, index=index)
            .sort_index()
            .unstack()[self.dataset_params["labels_columns"]]
        )

        # Rescale
        predictions = inverse_preprocess_data(
            data=predictions * self.dataset_params["winsorize_threshold"],
            scaler=self.label_scaler,
        )
        labels = inverse_preprocess_data(
            data=labels * self.dataset_params["winsorize_threshold"],
            scaler=self.label_scaler,
        )

        prediction_abs_bins = self._build_abs_bins(df=predictions)
        probability_bins = self._build_abs_bins(df=probabilities)

        # Store signals
        for data_type, data in [
            ("predictions", predictions),
            ("labels", labels),
            ("probabilities", probabilities),
            ("prediction_abs_bins", prediction_abs_bins),
            ("probability_bins", probability_bins),
        ]:
            to_parquet(
                df=data, path=os.path.join(save_dir, f"{data_type}.parquet.zstd"),
            )

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        id: Union[List, torch.Tensor],
        id_to_asset: Optional[Dict] = None,
    ):
        assert self.mode in ("predict")
        self.model.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        if not isinstance(id, torch.Tensor):
            id = torch.Tensor(id)

        data_dict = {"X": X.to(self.device), "ID": id.to(self.device).long()}
        x = self._build_stacked_features(data_dict=data_dict)

        (pred_abs_factor, pred_sign_factor, _, _,) = self.model(x=x, id=data_dict["ID"])

        preds = self._invert_to_prediction(
            pred_abs_factor=pred_abs_factor, pred_sign_factor=pred_sign_factor
        )
        predictions = pd.Series(preds.view(-1).cpu().tolist(), index=id.int().tolist(),)
        probabilities = pd.Series(
            self._build_probabilities(pred_sign_factor=pred_sign_factor)
            .view(-1)
            .cpu()
            .tolist(),
            index=id.int().tolist(),
        )

        # Post-process
        assert id_to_asset is not None
        predictions.index = predictions.index.map(lambda x: id_to_asset[x])
        probabilities.index = probabilities.index.map(lambda x: id_to_asset[x])

        # Rescale
        labels_columns = self.dataset_params["labels_columns"]
        labels_columns = [
            labels_column.replace("-", "/") for labels_column in labels_columns
        ]

        predictions = predictions.rename("predictions").to_frame().T[labels_columns]
        predictions = inverse_preprocess_data(
            data=predictions * self.dataset_params["winsorize_threshold"],
            scaler=self.label_scaler,
        ).loc["predictions"]

        probabilities = probabilities.rename("probabilities")[labels_columns]

        return {"predictions": predictions, "probabilities": probabilities}


if __name__ == "__main__":
    import fire

    fire.Fire(PredictorV1)
