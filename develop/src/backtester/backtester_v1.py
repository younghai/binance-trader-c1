import numpy as np
from .utils import data_loader, display_accuracy, Position

CONFIG = {
    "historical_pricing_path": "",
    "historical_predictions_path": "",
}


class BacktesterV1:
    def __init__(
        self,
        historical_pricing_path=CONFIG["historical_pricing_path"],
        historical_predictions_path=CONFIG["historical_predictions_path"],
    ):
        self.historical_pricing = data_loader(path=historical_pricing_path)

        tmp_data = data_loader(path=historical_predictions_path)
        self.historical_predictions = tmp_data["prediction"]
        self.historical_labels = tmp_data["label"]

    def display_accuracy(self):
        display_accuracy(
            historical_predictions=self.historical_predictions,
            historical_labels=self.historical_labels,
        )

    def run(self):
        pass
