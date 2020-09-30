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
    "exit_if_achieved": True,
    "achieved_with_commission": True,
    "achieved_with_aux_condition": False,
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
        exit_if_achieved=CONFIG["exit_if_achieved"],
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
        self.exit_if_achieved = exit_if_achieved
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
