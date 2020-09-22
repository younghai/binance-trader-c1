import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import abstractmethod
from IPython.display import display, display_markdown
from .utils import data_loader, Position
from comman_utils import make_dirs
from collections import OrderedDict
import empyrical as emp


CONFIG = {
    "position_side": "long",
    "entry_ratio": 0.1,
    "commission": 0.0015,
    "min_holding_minutes": 2,
    "max_holding_minutes": 10,
    "compound_interest": True,
    "possible_in_debt": False,
    "q_threshold": 7,
    "report_store_dir": "../../storage/report/fwd_10m/v001",
}


class BasicBacktester:
    def __init__(
        self,
        base_currency,
        bins_path,
        historical_pricing_path,
        historical_predictions_path,
        report_store_dir=CONFIG["report_store_dir"],
        position_side=CONFIG["position"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
        q_threshold=CONFIG["q_threshold"],
    ):
        assert position_side in ("long", "short", "longshort")
        self.base_currency = base_currency
        self.report_store_dir = report_store_dir
        self.position_side = position_side
        self.entry_ratio = entry_ratio
        self.commission = commission
        self.min_holding_minutes = min_holding_minutes
        self.max_holding_minutes = max_holding_minutes
        self.compound_interest = compound_interest
        self.possible_in_debt = possible_in_debt
        self.q_threshold = q_threshold
        make_dirs([report_store_dir])

        # Load data
        self.bins = self.load_bins(bins_path)
        (
            self.historical_pricing,
            self.historical_predictions,
            self.historical_labels,
        ) = self.load_historical_data(
            base_currency=base_currency,
            historical_pricing_path=historical_pricing_path,
            historical_predictions_path=historical_predictions_path,
        )
        self.tradable_coins = self.historical_pricing.columns
        self.index = self.historical_predictions.index

        self.initialize()

    def load_tradable_coins(self, tradable_coins_path):
        with open("tradable_coins_path", "r") as f:
            tradable_coins = f.read().splitlines()

        return tradable_coins

    def load_bins(self, bins_path):
        return pd.read_csv(bins_path, header=0, index_col=0)

    def load_historical_data(
        self, base_currency, historical_pricing_path, historical_predictions_path
    ):
        historical_pricing = data_loader(path=historical_pricing_path)

        tmp_data = data_loader(path=historical_predictions_path)
        historical_predictions = tmp_data["prediction"]
        historical_labels = tmp_data["label"]

        # Re-order columns
        columns = historical_pricing.columns
        historical_predictions.columns = columns
        historical_labels.columns = columns

        # Filter by base_currency
        columns_with_base_currency = columns[
            columns.str.endswith(base_currency.upper())
        ]
        historical_pricing = historical_pricing[columns_with_base_currency]
        historical_predictions = historical_predictions[columns_with_base_currency]
        historical_labels = historical_labels[columns_with_base_currency]

        return historical_pricing, historical_predictions, historical_labels

    def initialize(self):
        self.historical_cache = {}
        self.historical_capital = {}

        self.positions = []
        self.cache = 1

    def report(self, value, target, now):
        assert now not in getattr(self, target)
        getattr(self, target)[now] = value

    def generate_report(self):
        historical_cache = pd.Series(self.historical_cache).rename("cache")
        historical_capital = pd.Series(self.historical_capital).rename("capital")
        historical_return = (
            pd.Series(self.historical_capital)
            .pct_change(fill_method=None)
            .fillna(0)
            .rename("return")
        )

        return pd.concat(
            [historical_cache, historical_capital, historical_return], axis=1
        )

    def store_report(self, report):
        report.to_csv(
            os.path.join(self.report_store_dir, f"report_{self.base_currency}.csv")
        )
        params = {
            "base_currency": self.base_currency,
            "position_side": self.position_side,
            "entry_ratio": self.entry_ratio,
            "commission": self.commission,
            "min_holding_minutes": self.min_holding_minutes,
            "max_holding_minutes": self.max_holding_minutes,
            "compound_interest": self.compound_interest,
            "possible_in_debt": self.possible_in_debt,
            "report_store_dir": self.report_store_dir,
            "tradable_coins": self.tradable_coins,
            "q_threshold": self.q_threshold,
        }
        with open(
            os.path.join(self.report_store_dir, f"params_{self.base_currency}.csv"), "w"
        ) as f:
            json.dump(params, f)

        print(f"[+] Report is stored: {self.base_currency}")

    def display_accuracy(self):
        accuracies = {}

        for column in self.historical_predictions.columns:
            class_accuracy = {}
            for class_num in range(self.historical_labels[column].max()):
                class_mask = self.historical_labels[column] == class_num
                class_accuracy["class_" + str(class_num)] = (
                    self.historical_predictions[column][class_mask] == class_num
                ).mean()

            accuracy = pd.Series(
                {
                    "total": (
                        self.historical_predictions[column]
                        == self.historical_labels[column]
                    ).mean(),
                    **class_accuracy,
                }
            )
            accuracies[column] = accuracy

        accuracies = pd.concat(accuracies).unstack().T
        display_markdown("#### Accuracy of signals", raw=True)
        display(accuracies)

    def display_metrics(self):
        assert len(self.historical_cache) != 0
        assert len(self.historical_capital) != 0

        historical_returns = (
            pd.Series(self.historical_capital).pct_change(fill_method=None).fillna(0)
        )

        metrics = OrderedDict()

        metrics["winning_ratio"] = (historical_returns > 0).mean()
        metrics["sharpe_ratio"] = emp.sharpe_ratio(historical_returns)
        metrics["max_drawdown"] = emp.max_drawdown(historical_returns)
        metrics["avg_return"] = historical_returns.mean()
        metrics["total_return"] = historical_returns.add(1).cumprod().sub(1).iloc[-1]

        display_markdown(f"#### Performance metrics: {self.base_currency}", raw=True)
        display(pd.Series(metrics))

    def display_report(self, report):
        display_markdown(f"#### Report: {self.base_currency}", raw=True)
        _, ax = plt.subplots(3, 1, figsize=(12, 9))

        for idx, column in enumerate(["capital", "return", "cache"]):
            report[column].plot(ax=ax[idx])
            ax[idx].set_title(f"historical {column}")

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def run(self):
        pass
