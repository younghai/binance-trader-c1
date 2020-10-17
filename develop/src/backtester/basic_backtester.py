import os
import json
import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
from abc import abstractmethod
from IPython.display import display, display_markdown
from .utils import data_loader, Position
from common_utils import make_dirs
from collections import OrderedDict, defaultdict
import empyrical as emp


CONFIG = {
    "report_prefix": "001",
    "position_side": "long",
    "entry_ratio": 0.1,
    "commission": 0.0015,
    "min_holding_minutes": 1,
    "max_holding_minutes": 10,
    "compound_interest": True,
    "possible_in_debt": False,
    "exit_if_achieved": True,
    "achieved_with_commission": False,
    "max_n_updated": None,
    "exit_q_threshold": 9,
}


class BasicBacktester:
    def __init__(
        self,
        base_currency,
        dataset_dir,
        exp_dir,
        report_prefix=CONFIG["report_prefix"],
        position_side=CONFIG["position_side"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
        exit_if_achieved=CONFIG["exit_if_achieved"],
        achieved_with_commission=CONFIG["achieved_with_commission"],
        max_n_updated=CONFIG["max_n_updated"],
        exit_q_threshold=CONFIG["exit_q_threshold"],
    ):
        assert position_side in ("long", "short", "longshort")
        self.base_currency = base_currency
        self.report_prefix = report_prefix
        self.position_side = position_side
        self.entry_ratio = entry_ratio
        self.commission = commission
        self.min_holding_minutes = min_holding_minutes
        self.max_holding_minutes = max_holding_minutes
        self.compound_interest = compound_interest
        self.possible_in_debt = possible_in_debt
        self.exit_if_achieved = exit_if_achieved
        self.achieved_with_commission = achieved_with_commission
        self.max_n_updated = max_n_updated

        # Set path to load data
        dataset_params_path = os.path.join(dataset_dir, "params.json")
        bins_path = os.path.join(dataset_dir, "bins.csv")

        self.report_store_dir = os.path.join(exp_dir, "reports/")
        make_dirs([self.report_store_dir])

        # Load data
        self.bins = self.load_bins(bins_path)
        dataset_params = self.load_dataset_params(dataset_params_path)

        if exit_q_threshold is None:
            self.exit_q_threshold = dataset_params["q_threshold"]
        else:
            self.exit_q_threshold = exit_q_threshold

        self.n_bins = dataset_params["n_bins"]
        self.historical_data_dict = self.build_historical_data_dict(
            base_currency=base_currency,
            historical_data_path_dict={
                "pricing": os.path.join(dataset_dir, "test/pricing.csv"),
                "predictions": os.path.join(
                    exp_dir, "generated_output/predictions.csv"
                ),
                "labels": os.path.join(exp_dir, "generated_output/labels.csv"),
                "q_predictions": os.path.join(
                    exp_dir, "generated_output/q_predictions.csv"
                ),
                "q_labels": os.path.join(exp_dir, "generated_output/q_labels.csv"),
            },
        )
        self.tradable_coins = self.historical_data_dict["pricing"].columns
        self.index = self.historical_data_dict["predictions"].index

        self.initialize()

    def load_tradable_coins(self, tradable_coins_path):
        with open(tradable_coins_path, "r") as f:
            tradable_coins = f.read().splitlines()

        return tradable_coins

    def load_bins(self, bins_path):
        return pd.read_csv(bins_path, header=0, index_col=0)

    def load_dataset_params(self, dataset_params_path):
        with open(dataset_params_path, "r") as f:
            dataset_params = json.load(f)

        return dataset_params

    def build_historical_data_dict(self, base_currency, historical_data_path_dict):
        historical_data_path_dict = copy(historical_data_path_dict)

        data_dict = {}
        data_dict["pricing"] = data_loader(
            path=historical_data_path_dict.pop("pricing"), compression="gzip"
        )
        columns = data_dict["pricing"].columns
        columns_with_base_currency = columns[
            columns.str.endswith(base_currency.upper())
        ]
        data_dict["pricing"] = data_dict["pricing"][columns_with_base_currency]

        for data_type, data_path in historical_data_path_dict.items():
            data_path = data_path.replace("aux_", "")
            data_dict[data_type] = data_loader(path=data_path)

            # Re-order columns
            data_dict[data_type].columns = columns

            # Filter by base_currency
            data_dict[data_type] = data_dict[data_type][columns_with_base_currency]

        return data_dict

    def initialize(self):
        self.historical_caches = {}
        self.historical_capitals = {}
        self.historical_entry_reasons = defaultdict(list)
        self.historical_exit_reasons = defaultdict(list)
        self.historical_profits = defaultdict(list)
        self.historical_trade_returns = defaultdict(list)
        self.historical_positions = {}

        self.positions = []
        self.cache = 1

    def report(self, value, target, now, append=False):
        if append is True:
            getattr(self, target)[now].append(value)
            return

        assert now not in getattr(self, target)
        getattr(self, target)[now] = value

    def generate_report(self):
        historical_caches = pd.Series(self.historical_caches).rename("cache")
        historical_capitals = pd.Series(self.historical_capitals).rename("capital")
        historical_returns = (
            pd.Series(self.historical_capitals)
            .pct_change(fill_method=None)
            .fillna(0)
            .rename("return")
        )
        historical_entry_reasons = pd.Series(self.historical_entry_reasons).rename(
            "entry_reason"
        )
        historical_exit_reasons = pd.Series(self.historical_exit_reasons).rename(
            "exit_reason"
        )
        historical_profits = pd.Series(self.historical_profits).rename("profit")
        historical_trade_returns = pd.Series(self.historical_trade_returns).rename(
            "trade_return"
        )
        historical_positions = pd.Series(self.historical_positions).rename("position")

        report = pd.concat(
            [
                historical_caches,
                historical_capitals,
                historical_returns,
                historical_entry_reasons,
                historical_exit_reasons,
                historical_profits,
                historical_trade_returns,
                historical_positions,
            ],
            axis=1,
        ).sort_index()
        report.index = pd.to_datetime(report.index)

        return report

    def store_report(self, report):
        report.to_csv(
            os.path.join(
                self.report_store_dir,
                f"report_{self.report_prefix}_{self.base_currency}.csv",
            )
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
            "achieved_with_commission": self.achieved_with_commission,
            "max_n_updated": self.max_n_updated,
            "tradable_coins": tuple(self.tradable_coins.tolist()),
            "exit_q_threshold": self.exit_q_threshold,
        }
        with open(
            os.path.join(
                self.report_store_dir,
                f"params_{self.report_prefix}_{self.base_currency}.csv",
            ),
            "w",
        ) as f:
            json.dump(params, f)

        print(f"[+] Report is stored: {self.report_prefix}_{self.base_currency}")

    def display_accuracy(self, predictions, labels):
        accuracies = {}

        for column in predictions.columns:
            class_accuracy = {}
            for class_num in range(labels[column].max() + 1):
                class_mask = predictions[column] == class_num
                class_accuracy["class_" + str(class_num)] = (
                    labels[column][class_mask] == class_num
                ).mean()

            accuracy = pd.Series(
                {
                    "total": (predictions[column] == labels[column]).mean(),
                    **class_accuracy,
                }
            )
            accuracies[column] = accuracy

        accuracies = pd.concat(accuracies).unstack().T
        display_markdown("#### Accuracy of signals", raw=True)
        display(accuracies)

    def display_metrics(self):
        assert len(self.historical_caches) != 0
        assert len(self.historical_capitals) != 0
        assert len(self.historical_trade_returns) != 0

        historical_returns = (
            pd.Series(self.historical_capitals).pct_change(fill_method=None).fillna(0)
        )
        historical_trade_returns = (
            pd.Series(self.historical_trade_returns).dropna().apply(lambda x: sum(x))
        )

        metrics = OrderedDict()
        metrics["trade_winning_ratio"] = (
            historical_trade_returns[historical_trade_returns != 0] > 0
        ).mean()
        metrics["trade_sharpe_ratio"] = emp.sharpe_ratio(historical_trade_returns)

        metrics["winning_ratio"] = (
            historical_returns[historical_returns != 0] > 0
        ).mean()
        metrics["sharpe_ratio"] = emp.sharpe_ratio(historical_returns)
        metrics["max_drawdown"] = emp.max_drawdown(historical_returns)
        metrics["avg_return"] = historical_returns.mean()
        metrics["total_return"] = historical_returns.add(1).cumprod().sub(1).iloc[-1]
        display_markdown(f"#### Performance metrics: {self.base_currency}", raw=True)
        display(pd.Series(metrics))

    def display_report(self, report):
        display_markdown(f"#### Report: {self.base_currency}", raw=True)
        _, ax = plt.subplots(4, 1, figsize=(12, 12))

        for idx, column in enumerate(["capital", "cache", "return", "trade_return"]):
            if column == "trade_return":
                report[column].dropna().apply(lambda x: sum(x)).plot(ax=ax[idx])

            else:
                report[column].plot(ax=ax[idx])

            ax[idx].set_title(f"historical {column}")

        plt.tight_layout()
        plt.show()

    def compute_cost_to_order(self, position):
        cache_to_order = position.entry_price * position.qty
        commission_to_order = cache_to_order * self.commission

        return cache_to_order + commission_to_order

    def check_if_executable_order(self, cost):
        if self.possible_in_debt is True:
            return True

        return bool((self.cache - cost) >= 0)

    def pay_cache(self, cost):
        self.cache = self.cache - cost

    def deposit_cache(self, profit):
        self.cache = self.cache + profit

    def update_position_if_already_have(self, position):
        for idx, exist_position in enumerate(self.positions):
            if (exist_position.asset == position.asset) and (
                exist_position.side == position.side
            ):
                # Skip when position has max_n_updated
                if self.max_n_updated is not None:
                    if exist_position.n_updated == self.max_n_updated:

                        # Update only entry_at
                        update_position = Position(
                            asset=exist_position.asset,
                            side=exist_position.side,
                            qty=exist_position.qty,
                            entry_price=exist_position.entry_price,
                            entry_at=position.entry_at,
                            n_updated=exist_position.n_updated,
                        )

                        self.positions[idx] = update_position
                        # return fake updated mark
                        return True

                update_entry_price = (
                    (exist_position.entry_price * exist_position.qty)
                    + (position.entry_price * position.qty)
                ) / (exist_position.qty + position.qty)

                # Update entry_price, entry_at and qty
                update_position = Position(
                    asset=exist_position.asset,
                    side=exist_position.side,
                    qty=exist_position.qty + position.qty,
                    entry_price=update_entry_price,
                    entry_at=position.entry_at,
                    n_updated=exist_position.n_updated + 1,
                )

                # Compute cost by only current order
                cost = self.compute_cost_to_order(position=position)
                executable_order = self.check_if_executable_order(cost=cost)

                # Update
                if executable_order is True:
                    self.pay_cache(cost=cost)
                    self.positions[idx] = update_position

                    # updated
                    return True

        return False

    def compute_profit(self, position, pricing, now):
        current_price = pricing[position.asset]

        if position.side == "long":
            profit_without_commission = current_price * position.qty

        if position.side == "short":
            profit_without_commission = position.entry_price * position.qty
            profit_without_commission += (
                (current_price - position.entry_price) * position.qty * -1
            )

        commission_to_order = profit_without_commission * self.commission

        return profit_without_commission - commission_to_order

    def compute_capital(self, pricing, now):
        # capital = cache + value of positions
        capital = self.cache

        for position in self.positions:
            current_price = pricing[position.asset]

            if position.side == "long":
                capital += current_price * position.qty

            if position.side == "short":
                capital += position.entry_price * position.qty
                capital += (current_price - position.entry_price) * position.qty * -1

        return capital

    def check_if_opposite_position_exists(self, order_asset, order_side):
        if order_side == "long":
            opposite_side = "short"
        if order_side == "short":
            opposite_side = "long"

        for exist_position in self.positions:
            if (exist_position.asset == order_asset) and (
                exist_position.side == opposite_side
            ):
                return True

        return False

    def entry_order(self, asset, side, cache_to_order, pricing, now):
        if cache_to_order == 0:
            return

        # if opposite position exists, we dont entry
        if (
            self.check_if_opposite_position_exists(order_asset=asset, order_side=side)
            is True
        ):
            return

        entry_price = pricing[asset]
        qty = cache_to_order / entry_price

        position = Position(
            asset=asset, side=side, qty=qty, entry_price=entry_price, entry_at=now
        )

        updated = self.update_position_if_already_have(position=position)
        if updated is True:
            self.report(
                value={asset: "updated"},
                target="historical_entry_reasons",
                now=now,
                append=True,
            )
            return
        else:
            cost = self.compute_cost_to_order(position=position)
            executable_order = self.check_if_executable_order(cost=cost)

            if executable_order is True:
                self.pay_cache(cost=cost)
                self.positions.append(position)
                self.report(
                    value={asset: "signal"},
                    target="historical_entry_reasons",
                    now=now,
                    append=True,
                )

    def exit_order(self, position, pricing, now):
        profit = self.compute_profit(position=position, pricing=pricing, now=now)
        self.deposit_cache(profit=profit)

        net_profit = profit - (position.entry_price * position.qty)
        self.report(value=net_profit, target="historical_profits", now=now, append=True)
        self.report(
            value=(net_profit / (position.entry_price * position.qty)),
            target="historical_trade_returns",
            now=now,
            append=True,
        )

    def handle_entry(
        self, cache_to_order, positive_assets, negative_assets, pricing, now
    ):
        # Entry order
        if self.position_side in ("long", "longshort"):
            for order_asset in positive_assets:
                self.entry_order(
                    asset=order_asset,
                    side="long",
                    cache_to_order=cache_to_order,
                    pricing=pricing,
                    now=now,
                )

        if self.position_side in ("short", "longshort"):
            for order_asset in negative_assets:
                self.entry_order(
                    asset=order_asset,
                    side="short",
                    cache_to_order=cache_to_order,
                    pricing=pricing,
                    now=now,
                )

    def handle_exit(self, positive_assets, negative_assets, pricing, now):
        for position_idx, position in enumerate(self.positions):
            passed_minutes = (
                pd.Timestamp(now) - pd.Timestamp(position.entry_at)
            ).total_seconds() / 60

            # Handle min_holding_minutes
            if passed_minutes <= self.min_holding_minutes:
                continue

            # Handle max_holding_minutes
            if passed_minutes >= self.max_holding_minutes:
                self.exit_order(position=position, pricing=pricing, now=now)
                self.report(
                    value={position.asset: "max_holding_minutes"},
                    target="historical_exit_reasons",
                    now=now,
                    append=True,
                )
                self.positions[position_idx].is_exited = True
                continue

            # Handle exit signal
            if (position.side == "long") and (position.asset in negative_assets):
                self.exit_order(position=position, pricing=pricing, now=now)
                self.report(
                    value={position.asset: "opposite_signal"},
                    target="historical_exit_reasons",
                    now=now,
                    append=True,
                )
                self.positions[position_idx].is_exited = True
                continue

            if (position.side == "short") and (position.asset in positive_assets):
                self.exit_order(position=position, pricing=pricing, now=now)
                self.report(
                    value={position.asset: "opposite_signal"},
                    target="historical_exit_reasons",
                    now=now,
                    append=True,
                )
                self.positions[position_idx].is_exited = True
                continue

            # Handle achievement
            if self.exit_if_achieved is True:
                if (
                    self.check_if_achieved(position=position, pricing=pricing, now=now)
                    is True
                ):
                    self.exit_order(position=position, pricing=pricing, now=now)
                    self.report(
                        value={position.asset: "achieved"},
                        target="historical_exit_reasons",
                        now=now,
                        append=True,
                    )
                    self.positions[position_idx].is_exited = True
                    continue

        # Delete exited positions
        self.positions = [
            position for position in self.positions if position.is_exited is not True
        ]

    @abstractmethod
    def check_if_achieved(self, position, pricing, now):
        pass

    @abstractmethod
    def run(self):
        pass
