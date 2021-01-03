import os
import json
import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
from abc import abstractmethod
from IPython.display import display, display_markdown
from .utils import load_parquet, Position
from common_utils_dev import make_dirs
from collections import OrderedDict, defaultdict
import empyrical as emp
from common_utils_dev import to_parquet


CONFIG = {
    "report_prefix": "001",
    "detail_report": False,
    "position_side": "longshort",
    "entry_ratio": 0.055,
    "commission": {"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
    "min_holding_minutes": 1,
    "max_holding_minutes": 30,
    "compound_interest": True,
    "order_criterion": "capital",
    "possible_in_debt": False,
    "exit_if_achieved": True,
    "achieve_ratio": 1,
    "achieved_with_commission": False,
    "max_n_updated": 0,
    "entry_threshold": 8,
    "exit_threshold": "auto",
    "probability_threshold": 0.2,
    "adjust_prediction": False,
}


def make_flat(series):
    flatten = []
    for key, values in series.to_dict().items():
        if isinstance(values, list):
            for value in values:
                flatten.append(pd.Series({key: value}))
        else:
            flatten.append(pd.Series({key: values}))

    return pd.concat(flatten).sort_index()


class BasicBacktester:
    def __init__(
        self,
        base_currency,
        dataset_dir,
        exp_dir,
        report_prefix=CONFIG["report_prefix"],
        detail_report=CONFIG["detail_report"],
        position_side=CONFIG["position_side"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        order_criterion=CONFIG["order_criterion"],
        possible_in_debt=CONFIG["possible_in_debt"],
        exit_if_achieved=CONFIG["exit_if_achieved"],
        achieve_ratio=CONFIG["achieve_ratio"],
        achieved_with_commission=CONFIG["achieved_with_commission"],
        max_n_updated=CONFIG["max_n_updated"],
        entry_threshold=CONFIG["entry_threshold"],
        exit_threshold=CONFIG["exit_threshold"],
        probability_threshold=CONFIG["probability_threshold"],
        adjust_prediction=CONFIG["adjust_prediction"],
    ):
        assert position_side in ("long", "short", "longshort")
        self.base_currency = base_currency
        self.report_prefix = report_prefix
        self.detail_report = detail_report
        self.position_side = position_side
        self.entry_ratio = entry_ratio
        self.commission = commission
        self.min_holding_minutes = min_holding_minutes
        self.max_holding_minutes = max_holding_minutes
        self.compound_interest = compound_interest
        self.order_criterion = order_criterion
        assert self.order_criterion in ("cache", "capital")

        self.possible_in_debt = possible_in_debt
        self.exit_if_achieved = exit_if_achieved
        self.achieve_ratio = achieve_ratio
        self.achieved_with_commission = achieved_with_commission
        self.max_n_updated = max_n_updated
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        assert isinstance(exit_threshold, (float, int, str))
        if type(exit_threshold) == str:
            assert (exit_threshold == "auto") or ("*" in exit_threshold)

        self.probability_threshold = probability_threshold
        assert (probability_threshold >= 0) and (probability_threshold <= 1)

        self.adjust_prediction = adjust_prediction

        self.dataset_dir = dataset_dir
        self.exp_dir = exp_dir

        self.initialize()

    def _load_prediction_abs_bins(self):
        return load_parquet(
            path=os.path.join(
                self.exp_dir, "generated_output/prediction_abs_bins.parquet.zstd"
            )
        )

    def _build_historical_data_dict(self, base_currency, historical_data_path_dict):
        historical_data_path_dict = copy(historical_data_path_dict)

        data_dict = {}

        # We use open pricing to handling, entry: open, exit: open
        data_dict["pricing"] = (
            load_parquet(path=historical_data_path_dict.pop("pricing"))
            .xs("open", axis=1, level=1)
            .astype("float16")
        )
        columns = data_dict["pricing"].columns
        columns_with_base_currency = columns[
            columns.str.endswith(base_currency.upper())
        ]
        data_dict["pricing"] = data_dict["pricing"][columns_with_base_currency]

        for data_type, data_path in historical_data_path_dict.items():
            data_dict[data_type] = load_parquet(path=data_path).astype("float16")

            # Filter by base_currency
            data_dict[data_type] = data_dict[data_type][columns_with_base_currency]

        return data_dict

    def _set_bins(self, prediction_abs_bins, index):
        assert (prediction_abs_bins >= 0).all().all()

        self.entry_bins = None
        self.exit_bins = None

        if isinstance(self.entry_threshold, str):
            if "*" in self.entry_threshold:
                self.entry_bins = (
                    prediction_abs_bins.loc[int(self.entry_threshold.split("*")[0])]
                    * float(self.entry_threshold.split("*")[-1])
                )[index]
        else:
            self.entry_bins = prediction_abs_bins.loc[self.entry_threshold][index]

        if isinstance(self.exit_threshold, str):
            if "*" in self.exit_threshold:
                self.exit_bins = (
                    prediction_abs_bins.loc[int(self.exit_threshold.split("*")[0])]
                    * float(self.exit_threshold.split("*")[-1])
                )[index]
        else:
            self.exit_bins = prediction_abs_bins.loc[self.exit_threshold][index]

    def build(self):
        self.report_store_dir = os.path.join(self.exp_dir, "reports/")
        make_dirs([self.report_store_dir])

        self.historical_data_dict = self._build_historical_data_dict(
            base_currency=self.base_currency,
            historical_data_path_dict={
                "pricing": os.path.join(self.dataset_dir, "test/pricing.parquet.zstd"),
                "predictions": os.path.join(
                    self.exp_dir, "generated_output/predictions.parquet.zstd"
                ),
                "probabilities": os.path.join(
                    self.exp_dir, "generated_output/probabilities.parquet.zstd"
                ),
                "labels": os.path.join(
                    self.exp_dir, "generated_output/labels.parquet.zstd"
                ),
            },
        )
        self.tradable_coins = self.historical_data_dict["predictions"].columns
        self.index = (
            self.historical_data_dict["predictions"].index
            & self.historical_data_dict["pricing"].index
        ).sort_values()
        for key in self.historical_data_dict.keys():
            self.historical_data_dict[key] = self.historical_data_dict[key].reindex(
                self.index
            )

        prediction_abs_bins = self._load_prediction_abs_bins()
        self._set_bins(
            prediction_abs_bins=prediction_abs_bins, index=self.tradable_coins
        )

    def initialize(self):
        self.historical_caches = {}
        self.historical_capitals = {}
        self.historical_trade_returns = defaultdict(list)

        if self.detail_report is True:
            self.historical_entry_reasons = defaultdict(list)
            self.historical_exit_reasons = defaultdict(list)
            self.historical_profits = defaultdict(list)
            self.historical_positions = {}

        self.positions = []
        self.cache = 1

    def report(self, value, target, now, append=False):
        if hasattr(self, target) is False:
            return

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
        historical_trade_returns = pd.Series(self.historical_trade_returns).rename(
            "trade_return"
        )

        report = [
            historical_caches,
            historical_capitals,
            historical_returns,
            historical_trade_returns,
        ]

        if self.detail_report is True:
            historical_entry_reasons = pd.Series(self.historical_entry_reasons).rename(
                "entry_reason"
            )
            historical_exit_reasons = pd.Series(self.historical_exit_reasons).rename(
                "exit_reason"
            )
            historical_profits = pd.Series(self.historical_profits).rename("profit")
            historical_positions = pd.Series(self.historical_positions).rename(
                "position"
            )

            report += [
                historical_entry_reasons,
                historical_exit_reasons,
                historical_profits,
                historical_positions,
            ]

        report = pd.concat(report, axis=1).sort_index()
        report.index = pd.to_datetime(report.index)

        return report

    def store_report(self, report):
        metrics = self.build_metrics().to_frame().T
        to_parquet(
            df=metrics.astype("float32"),
            path=os.path.join(
                self.report_store_dir,
                f"metrics_{self.report_prefix}_{self.base_currency}.parquet.zstd",
            ),
        )

        to_parquet(
            df=report,
            path=os.path.join(
                self.report_store_dir,
                f"report_{self.report_prefix}_{self.base_currency}.parquet.zstd",
            ),
        )

        params = {
            "base_currency": self.base_currency,
            "position_side": self.position_side,
            "entry_ratio": self.entry_ratio,
            "commission": self.commission,
            "min_holding_minutes": self.min_holding_minutes,
            "max_holding_minutes": self.max_holding_minutes,
            "compound_interest": self.compound_interest,
            "order_criterion": self.order_criterion,
            "possible_in_debt": self.possible_in_debt,
            "achieved_with_commission": self.achieved_with_commission,
            "max_n_updated": self.max_n_updated,
            "tradable_coins": tuple(self.tradable_coins.tolist()),
            "exit_if_achieved": self.exit_if_achieved,
            "achieve_ratio": self.achieve_ratio,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "probability_threshold": self.probability_threshold,
            "adjust_prediction": self.adjust_prediction,
        }
        with open(
            os.path.join(
                self.report_store_dir,
                f"params_{self.report_prefix}_{self.base_currency}.json",
            ),
            "w",
        ) as f:
            json.dump(params, f)

        print(f"[+] Report is stored: {self.report_prefix}_{self.base_currency}")

    def build_metrics(self):
        assert len(self.historical_caches) != 0
        assert len(self.historical_capitals) != 0
        assert len(self.historical_trade_returns) != 0

        historical_returns = (
            pd.Series(self.historical_capitals).pct_change(fill_method=None).fillna(0)
        )
        historical_trade_returns = make_flat(
            pd.Series(self.historical_trade_returns).rename("trade_return").dropna()
        )

        metrics = OrderedDict()
        metrics["trade_winning_ratio"] = (
            historical_trade_returns[historical_trade_returns != 0] > 0
        ).mean()
        metrics["trade_sharpe_ratio"] = emp.sharpe_ratio(historical_trade_returns)
        metrics["trade_avg_return"] = historical_trade_returns.mean()
        metrics["max_drawdown"] = emp.max_drawdown(historical_returns)
        metrics["total_return"] = historical_returns.add(1).cumprod().sub(1).iloc[-1]

        return pd.Series(metrics)

    def display_metrics(self):
        display_markdown(f"#### Performance metrics: {self.base_currency}", raw=True)
        display(self.build_metrics())

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
        commission_to_order = cache_to_order * (
            self.commission["entry"] + self.commission["spread"]
        )

        return cache_to_order + commission_to_order

    def check_if_executable_order(self, cost):
        if self.possible_in_debt is True:
            return True

        return bool((self.cache - cost) >= 0)

    def pay_cache(self, cost):
        self.cache = self.cache - cost

    def deposit_cache(self, profit):
        self.cache = self.cache + profit

    def compute_adjusted_prediction(
        self, side, entry_price, current_price, entry_prediction, current_prediction
    ):
        if side == "long":
            if entry_price * (1 + entry_prediction) < current_price * (
                1 + current_prediction
            ):
                return ((current_price * (1 + current_prediction)) / entry_price) - 1

        if side == "short":
            if entry_price * (1 + entry_prediction) > current_price * (
                1 + current_prediction
            ):
                return ((current_price * (1 + current_prediction)) / entry_price) - 1

        return entry_prediction

    def update_position_if_already_have(self, position):
        for idx, exist_position in enumerate(self.positions):
            if (exist_position.asset == position.asset) and (
                exist_position.side == position.side
            ):
                # Skip when position has max_n_updated
                if self.max_n_updated is not None:
                    if exist_position.n_updated == self.max_n_updated:

                        adjusted_prediction = exist_position.prediction
                        if self.adjust_prediction is True:
                            adjusted_prediction = self.compute_adjusted_prediction(
                                side=exist_position.side,
                                entry_price=exist_position.entry_price,
                                current_price=position.entry_price,
                                entry_prediction=exist_position.prediction,
                                current_prediction=position.prediction,
                            )

                        # Update only prediction, and entry_at
                        update_position = Position(
                            asset=exist_position.asset,
                            side=exist_position.side,
                            qty=exist_position.qty,
                            entry_price=exist_position.entry_price,
                            prediction=adjusted_prediction,
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

                # This is currently invalid way, but acceptable.
                update_prediction = (
                    (exist_position.prediction * exist_position.qty)
                    + (position.prediction * position.qty)
                ) / (exist_position.qty + position.qty)

                # Update entry_price, entry_at and qty
                update_position = Position(
                    asset=exist_position.asset,
                    side=exist_position.side,
                    qty=exist_position.qty + position.qty,
                    entry_price=update_entry_price,
                    entry_at=position.entry_at,
                    prediction=update_prediction,
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

    def compute_profit(self, position, pricing, now, achieved=False):
        current_price = pricing[position.asset]

        if position.side == "long":
            profit_without_commission = current_price * position.qty

        if position.side == "short":
            profit_without_commission = position.entry_price * position.qty
            profit_without_commission += (
                (current_price - position.entry_price) * position.qty * -1
            )

        exit_commission = self.commission["exit"]
        if achieved is not True:
            exit_commission = (self.commission["exit"] * 2) + self.commission["spread"]

        commission_to_order = profit_without_commission * exit_commission

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

    def entry_order(self, asset, side, cache_to_order, pricing, prediction, now):
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
            asset=asset,
            side=side,
            qty=qty,
            entry_price=entry_price,
            prediction=prediction,
            entry_at=now,
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

    def exit_order(self, position, pricing, now, achieved=False):
        profit = self.compute_profit(
            position=position, pricing=pricing, now=now, achieved=achieved
        )
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
        self,
        predictions,
        cache_to_order,
        positive_assets,
        negative_assets,
        pricing,
        now,
    ):
        # Entry order
        if self.position_side in ("long", "longshort"):
            for order_asset in positive_assets:
                self.entry_order(
                    asset=order_asset,
                    side="long",
                    cache_to_order=cache_to_order,
                    pricing=pricing,
                    prediction=predictions[order_asset],
                    now=now,
                )

        if self.position_side in ("short", "longshort"):
            for order_asset in negative_assets:
                self.entry_order(
                    asset=order_asset,
                    side="short",
                    cache_to_order=cache_to_order,
                    pricing=pricing,
                    prediction=predictions[order_asset],
                    now=now,
                )

    def handle_exit(self, positive_assets, negative_assets, pricing, now):
        for position_idx, position in enumerate(self.positions):
            # Handle achievement
            if self.exit_if_achieved is True:
                if (
                    self.check_if_achieved(position=position, pricing=pricing, now=now)
                    is True
                ):
                    self.exit_order(
                        position=position, pricing=pricing, now=now, achieved=True
                    )
                    self.report(
                        value={position.asset: "achieved"},
                        target="historical_exit_reasons",
                        now=now,
                        append=True,
                    )
                    self.positions[position_idx].is_exited = True
                    continue

            # Keep position if matched
            if (position.side == "long") and (position.asset in positive_assets):
                continue

            if (position.side == "short") and (position.asset in negative_assets):
                continue

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

        # Delete exited positions
        self.positions = [
            position for position in self.positions if position.is_exited is not True
        ]

    def check_if_achieved(self, position, pricing, now):
        current_price = pricing[position.asset]

        diff_price = current_price - position.entry_price
        if self.achieved_with_commission is True:
            if position.side == "long":
                commission = (
                    current_price
                    * (self.commission["exit"] + self.commission["spread"])
                ) + (
                    position.entry_price
                    * (self.commission["entry"] + self.commission["spread"])
                )
            if position.side == "short":
                commission = -(
                    (
                        current_price
                        * (self.commission["exit"] + self.commission["spread"])
                    )
                    + (
                        position.entry_price
                        * (self.commission["entry"] + self.commission["spread"])
                    )
                )

            diff_price = diff_price - commission

        if diff_price != 0:
            trade_return = diff_price / position.entry_price
        else:
            trade_return = 0

        trade_return = trade_return / self.achieve_ratio

        if self.exit_threshold == "auto":
            if position.side == "long":
                assert position.prediction > 0
                if trade_return >= position.prediction:
                    return True

            if position.side == "short":
                assert position.prediction < 0
                if trade_return <= position.prediction:
                    return True
        else:
            if position.side == "long":
                if trade_return >= self.exit_bins[position.asset]:
                    return True

            if position.side == "short":
                if trade_return <= -self.exit_bins[position.asset]:
                    return True

        return False

    @abstractmethod
    def run(self):
        pass
