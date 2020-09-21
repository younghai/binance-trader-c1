import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from IPython.display import display
from .utils import data_loader, display_accuracy, Position
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
    "report_store_dir": "../../storage/report/fwd_10m/v001",
}


class BasicBacktester:
    def __init__(
        self,
        historical_pricing_path,
        historical_predictions_path,
        position_side=CONFIG["position"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
        report_store_dir=CONFIG["report_store_dir"],
    ):
        assert position_side in ("long", "short", "longshort")
        self.position_side = position_side
        self.entry_ratio = entry_ratio
        self.commission = commission
        self.min_holding_minutes = min_holding_minutes
        self.max_holding_minutes = max_holding_minutes
        self.compound_interest = compound_interest
        self.possible_in_debt = possible_in_debt
        self.report_store_dir = report_store_dir
        make_dirs([report_store_dir])

        (
            self.historical_pricing,
            self.historical_predictions,
            self.historical_labels,
        ) = self.build_historical_data(
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

    def build_historical_data(
        self, historical_pricing_path, historical_predictions_path
    ):
        historical_pricing = data_loader(path=historical_pricing_path)

        tmp_data = data_loader(path=historical_predictions_path)
        historical_predictions = tmp_data["prediction"]
        historical_labels = tmp_data["label"]

        historical_predictions.columns = historical_pricing.columns
        historical_labels.columns = historical_pricing.columns

        return historical_pricing, historical_predictions, historical_labels

    def initialize(self):
        self.historical_cache = {}
        self.historical_capital = {}

        self.positions = []
        self.cache = 1

    def report(self, value, target, now):
        assert now not in getattr(self, target)
        getattr(self, target)[now] = value

    def store_report(self):
        historical_cache = pd.Series(self.historical_cache).rename("cache")
        historical_capital = pd.Series(self.historical_capital).rename("capital")
        historical_return = (
            pd.Series(self.historical_capital)
            .pct_change(fill_method=None)
            .fillna(0)
            .rename("return")
        )

        pd.concat(
            [historical_cache, historical_capital, historical_return], axis=1
        ).to_csv(os.path.join(self.report_store_dir, "report.csv"))
        print("[+] Stored report")

    def display_accuracy(self):
        display_accuracy(
            historical_predictions=self.historical_predictions,
            historical_labels=self.historical_labels,
        )

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

        display(pd.Series(metrics))

    @abstractmethod
    def run(self):
        pass
