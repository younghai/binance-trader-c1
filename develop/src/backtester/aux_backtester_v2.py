import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from .utils import data_loader, Position, compute_quantile, nan_to_zero
from common_utils import make_dirs
from .basic_backtester import BasicBacktester
from tqdm import tqdm
from IPython.display import display_markdown, display


CONFIG = {
    "report_prefix": "001",
    "position_side": "long",
    "entry_ratio": 0.05,
    "commission": 0.001,
    "min_holding_minutes": 1,
    "max_holding_minutes": 10,
    "compound_interest": False,
    "possible_in_debt": True,
    "achieved_with_commission": False,
    "achieved_with_aux_condition": True,
    "max_n_updated": None,
    "entry_q_prediction_threhold": 8,
    "entry_aux_q_prediction_threhold": 8,
}


class AUXBacktesterV2(BasicBacktester):
    def __init__(
        self,
        base_currency,
        dataset_dir,
        exp_dir,
        aux_dataset_dir,
        aux_exp_dir,
        report_prefix=CONFIG["report_prefix"],
        position_side=CONFIG["position_side"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
        achieved_with_commission=CONFIG["achieved_with_commission"],
        achieved_with_aux_condition=CONFIG["achieved_with_aux_condition"],
        max_n_updated=CONFIG["max_n_updated"],
        entry_q_prediction_threhold=CONFIG["entry_q_prediction_threhold"],
        entry_aux_q_prediction_threhold=CONFIG["entry_aux_q_prediction_threhold"],
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
        self.achieved_with_commission = achieved_with_commission
        self.achieved_with_aux_condition = achieved_with_aux_condition
        self.max_n_updated = max_n_updated

        self.report_store_dir = os.path.join(exp_dir, "aux_reports/")
        make_dirs([self.report_store_dir])

        # Load data
        self.bins = self.load_bins(os.path.join(dataset_dir, "bins.csv"))
        self.aux_bins = self.load_bins(os.path.join(aux_dataset_dir, "bins.csv"))

        dataset_params = self.load_dataset_params(
            os.path.join(dataset_dir, "params.json")
        )
        self.q_threshold = dataset_params["q_threshold"]
        self.n_bins = dataset_params["n_bins"]

        aux_dataset_params = self.load_dataset_params(
            os.path.join(aux_dataset_dir, "params.json")
        )
        self.aux_q_threshold = aux_dataset_params["q_threshold"]
        self.aux_n_bins = aux_dataset_params["n_bins"]

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
                "aux_predictions": os.path.join(
                    aux_exp_dir, "generated_output/predictions.csv"
                ),
                "aux_labels": os.path.join(
                    aux_exp_dir, "generated_output/aux_labels.csv"
                ),
                "aux_q_predictions": os.path.join(
                    aux_exp_dir, "generated_output/q_predictions.csv"
                ),
                "aux_q_labels": os.path.join(
                    aux_exp_dir, "generated_output/q_labels.csv"
                ),
            },
        )
        self.tradable_coins = self.historical_data_dict["pricing"].columns
        self.index = (
            self.historical_data_dict["predictions"].index
            & self.historical_data_dict["aux_predictions"].index
        )

        self.entry_q_prediction_threhold = entry_q_prediction_threhold
        self.entry_aux_q_prediction_threhold = entry_aux_q_prediction_threhold

        self.initialize()

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
                        # return fake updated mark
                        return True

                update_entry_price = (
                    (exist_position.entry_price * exist_position.qty)
                    + (position.entry_price * position.qty)
                ) / (exist_position.qty + position.qty)

                # Update entry_at and qty
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

        commission_to_order = (current_price * position.qty) * self.commission

        return profit_without_commission - commission_to_order

    def check_if_achieved(self, position, pricing, now):
        current_price = pricing[position.asset]

        diff_price = current_price - position.entry_price
        if self.achieved_with_commission is True:
            if position.side == "long":
                commission = (current_price + position.entry_price) * self.commission
            if position.side == "short":
                commission = -((current_price + position.entry_price) * self.commission)

            diff_price = diff_price - commission

        if diff_price != 0:
            trade_return = diff_price / position.entry_price
        else:
            trade_return = 0

        if self.achieved_with_aux_condition is True:
            q = compute_quantile(trade_return, bins=self.aux_bins[position.asset])

            if position.side == "long":
                if q >= self.aux_q_threshold:
                    return True

            if position.side == "short":
                if q <= ((self.aux_n_bins - 1) - self.aux_q_threshold):
                    return True
        else:
            q = compute_quantile(trade_return, bins=self.bins[position.asset])

            if position.side == "long":
                if q >= self.q_threshold:
                    return True

            if position.side == "short":
                if q <= ((self.n_bins - 1) - self.q_threshold):
                    return True

        return False

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
            asset=asset,
            side=side,
            qty=qty,
            entry_price=entry_price,
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

    def exit_order(self, position, pricing, now):
        profit = self.compute_profit(position=position, pricing=pricing, now=now)
        self.deposit_cache(profit=profit)
        self.report(
            value={position.asset: profit},
            target="historical_profits",
            now=now,
            append=True,
        )

    def display_q_accuracy(self):
        accuracies = {}

        for column in self.historical_data_dict["q_predictions"].columns:
            class_accuracy = {}
            for class_num in range(
                self.historical_data_dict["q_labels"][column].max() + 1
            ):
                class_mask = (
                    self.historical_data_dict["q_predictions"][column] == class_num
                )
                class_accuracy["class_" + str(class_num)] = (
                    self.historical_data_dict["q_labels"][column][class_mask]
                    == class_num
                ).mean()

            accuracy = pd.Series(
                {
                    "total": (
                        self.historical_data_dict["q_predictions"][column]
                        == self.historical_data_dict["q_labels"][column]
                    ).mean(),
                    **class_accuracy,
                }
            )
            accuracies[column] = accuracy

        accuracies = pd.concat(accuracies).unstack().T
        display_markdown("#### Q Accuracy of signals", raw=True)
        display(accuracies)

    def run(self, display=True):
        self.initialize()

        for now in tqdm(self.index):
            # Step1: Prepare pricing and signal
            pricing = self.historical_data_dict["pricing"].loc[now]
            predictions = self.historical_data_dict["predictions"].loc[now]
            q_predictions = self.historical_data_dict["q_predictions"].loc[now]
            aux_predictions = self.historical_data_dict["aux_predictions"].loc[now]
            aux_q_predictions = self.historical_data_dict["aux_q_predictions"].loc[now]

            # Set assets which has signals
            positive_assets = self.tradable_coins[
                (predictions == 0)
                & (aux_predictions == 0)
                & (q_predictions >= self.entry_q_prediction_threhold)
                & (aux_q_predictions >= self.entry_aux_q_prediction_threhold)
            ]
            negative_assets = self.tradable_coins[
                (predictions == 1)
                & (aux_predictions == 1)
                & (
                    q_predictions
                    <= (self.n_bins - 1) - self.entry_q_prediction_threhold
                )
                & (
                    aux_q_predictions
                    <= (self.aux_n_bins - 1) - self.entry_aux_q_prediction_threhold
                )
            ]

            # Exit
            self.handle_exit(
                positive_assets=positive_assets,
                negative_assets=negative_assets,
                pricing=pricing,
                now=now,
            )

            # Compute how much use cache
            if self.compound_interest is False:
                cache_to_order = self.entry_ratio
            else:
                cache_to_order = nan_to_zero(value=(self.cache * self.entry_ratio))

            # Entry
            self.handle_entry(
                cache_to_order=cache_to_order,
                positive_assets=positive_assets,
                negative_assets=negative_assets,
                pricing=pricing,
                now=now,
            )

            # To report
            self.report(value=self.cache, target="historical_caches", now=now)
            self.report(
                value=self.compute_capital(pricing=pricing, now=now),
                target="historical_capitals",
                now=now,
            )
            self.report(
                value=self.positions,
                target="historical_positions",
                now=now,
            )

        report = self.generate_report()
        self.store_report(report=report)

        if display is True:
            display_markdown("#### Main Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["predictions"],
                labels=self.historical_data_dict["labels"],
            )

            display_markdown("#### Main Q Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["q_predictions"],
                labels=self.historical_data_dict["q_labels"],
            )

            display_markdown("#### Auxiliary Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["aux_predictions"],
                labels=self.historical_data_dict["aux_labels"],
            )

            display_markdown("#### Auxiliary Q Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["aux_q_predictions"],
                labels=self.historical_data_dict["aux_q_labels"],
            )

            self.display_metrics()
            self.display_report(report=report)


if __name__ == "__main__":
    import fire

    fire.Fire(AUXBacktesterV2)
