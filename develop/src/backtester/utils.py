import pandas as pd
from IPython.display import display


def data_loader(path):
    return pd.read_csv(path, header=0, index_col=0)


def display_accuracy(historical_predictions, historical_labels):
    total_accuracy = (historical_predictions == historical_labels).mean()

    class_accuracy = {}
    for class_num in range(historical_labels.max()):
        class_mask = historical_labels == class_num
        class_accuracy["class_" + str(class_num)] = (
            historical_predictions[class_mask] == class_num
        ).mean()

    accuracy = pd.Series({"total": total_accuracy, **class_accuracy})

    display(accuracy)


class Position:
    def __init__(self, asset, qty, entry_price, entry_at, base_currency):
        self.asset = asset
        self.qty = qty
        self.entry_price = entry_price
        self.entry_at = entry_at
        self.base_currency = base_currency

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"Position(asset={self.asset}, qty={self.qty}, entry_price={self.entry_price:.4f}, base_currency={self.base_currency})"

    def __str__(self):
        return self.__repr__()
