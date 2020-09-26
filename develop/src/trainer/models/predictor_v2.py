import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from .basic_predictor import BasicPredictor


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
    "criterion_params": {},
    "load_strict": False,
    "model_name": "BackboneV2",
    "model_params": {
        "in_channels": 320,
        "n_assets": 32,
        "n_class_y_per_asset": 4,
        "n_class_qy_per_asset": 10,
        "n_blocks": 3,
        "n_block_layers": 24,
        "growth_rate": 12,
        "dropout": 0.2,
        "channel_reduction": 0.5,
        "activation": "relu",
        "normalization": "bn",
        "seblock": True,
        "sablock": True,
    },
}


class PredictorV2(BasicPredictor):
    """
    Functions:
        train(): train the model with train_data
        generate(save_dir: str): generate predictions & labels with test_data
        predict(X: torch.Tensor): gemerate prediction with given data
    """

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
        super().__init__(
            data_dir=data_dir,
            test_data_dir=test_data_dir,
            d_config={**DATA_CONFIG, **d_config},
            m_config={**MODEL_CONFIG, **m_config},
            exp_dir=exp_dir,
            device=device,
            pin_memory=pin_memory,
            num_workers=num_workers,
            mode=mode,
        )

    def _compute_train_loss(self, train_data_dict):
        X, Y, QY = train_data_dict["X"], train_data_dict["Y"], train_data_dict["QY"]

        # Set train mode
        self.model.train()
        self.model.zero_grad()

        # Set loss
        y_preds, qy_preds = self.model(X)

        # Y loss
        y_preds_shape = y_preds.size()
        loss = self.criterion(y_preds.view(-1, y_preds_shape[-1]), Y.detach().view(-1))

        # QY loss
        qy_preds_shape = qy_preds.size()
        loss += self.criterion(
            qy_preds.view(-1, qy_preds_shape[-1]), QY.detach().view(-1)
        )

        return loss

    def _compute_test_loss(self, test_data_dict):
        X, Y, QY = test_data_dict["X"], test_data_dict["Y"], test_data_dict["QY"]

        # Set eval mode
        self.model.eval()

        # Set loss
        y_preds, qy_preds = self.model(X)

        # Y loss
        y_preds_shape = y_preds.size()
        loss = self.criterion(y_preds.view(-1, y_preds_shape[-1]), Y.detach().view(-1))

        # QY loss
        qy_preds_shape = qy_preds.size()
        loss += self.criterion(
            qy_preds.view(-1, qy_preds_shape[-1]), QY.detach().view(-1)
        )

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

    def generate(self, save_dir=None):
        assert self.mode in ("test")
        self.model.eval()

        if save_dir is None:
            save_dir = self.data_config["generate_output_dir"]

        index = self.test_data_loader.dataset.index

        predictions = []
        labels = []
        q_predictions = []
        q_labels = []
        for _ in tqdm(range(len(self.test_data_loader))):
            test_data_dict = self._generate_test_data_dict()

            X, Y, QY = test_data_dict["X"], test_data_dict["Y"], test_data_dict["QY"]

            y_preds, qy_preds = self.model(X)

            B, _, _ = y_preds.size()

            predictions += y_preds.argmax(dim=-1).view(B, -1).cpu().tolist()
            labels += Y.view(B, -1).cpu().tolist()

            q_predictions += qy_preds.argmax(dim=-1).view(B, -1).cpu().tolist()
            q_labels += QY.view(B, -1).cpu().tolist()

        pd.DataFrame(predictions, index=index).to_csv(
            os.path.join(save_dir, "predictions.csv")
        )
        pd.DataFrame(labels, index=index).to_csv(os.path.join(save_dir, "labels.csv"))

        pd.DataFrame(q_predictions, index=index).to_csv(
            os.path.join(save_dir, "q_predictions.csv")
        )
        pd.DataFrame(q_labels, index=index).to_csv(
            os.path.join(save_dir, "q_labels.csv")
        )

    def predict(self, X):
        self.model.eval()
        y_preds, qy_preds = self.model(X.to(self.device))

        return (y_preds.argmax(dim=-1).cpu(), qy_preds.argmax(dim=-1).cpu())


if __name__ == "__main__":
    import fire

    fire.Fire(PredictorV2)
