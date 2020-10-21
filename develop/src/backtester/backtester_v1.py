import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from .utils import data_loader, Position, compute_quantile, nan_to_zero
from .basic_backtester import BasicBacktester
from tqdm import tqdm
from IPython.display import display_markdown, display
import gc
import json
from common_utils import to_parquet


CONFIG = {
    "report_prefix": "001",
    "detail_report": False,
    "position_side": "long",
    "entry_ratio": 0.05,
    "commission": 0.0015,
    "min_holding_minutes": 1,
    "max_holding_minutes": 10,
    "compound_interest": False,
    "possible_in_debt": True,
    "exit_if_achieved": True,
    "achieved_with_commission": False,
    "max_n_updated": None,
    "entry_qay_threshold": 9,
    "entry_qby_threshold": 9,
    "entry_qay_prob_threshold": 0,
    "entry_qby_prob_threshold": 0,
    "exit_q_threshold": 9,
}


class BacktesterV1(BasicBacktester):
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
        possible_in_debt=CONFIG["possible_in_debt"],
        exit_if_achieved=CONFIG["exit_if_achieved"],
        achieved_with_commission=CONFIG["achieved_with_commission"],
        max_n_updated=CONFIG["max_n_updated"],
        exit_q_threshold=CONFIG["exit_q_threshold"],
        entry_qay_threshold=CONFIG["entry_qay_threshold"],
        entry_qby_threshold=CONFIG["entry_qby_threshold"],
        entry_qay_prob_threshold=CONFIG["entry_qay_prob_threshold"],
        entry_qby_prob_threshold=CONFIG["entry_qby_prob_threshold"],
    ):
        super().__init__(
            base_currency=base_currency,
            dataset_dir=dataset_dir,
            exp_dir=exp_dir,
            report_prefix=report_prefix,
            detail_report=detail_report,
            position_side=position_side,
            entry_ratio=entry_ratio,
            commission=commission,
            min_holding_minutes=min_holding_minutes,
            max_holding_minutes=max_holding_minutes,
            compound_interest=compound_interest,
            possible_in_debt=possible_in_debt,
            exit_if_achieved=exit_if_achieved,
            achieved_with_commission=achieved_with_commission,
            max_n_updated=max_n_updated,
            exit_q_threshold=exit_q_threshold,
        )

        self.entry_qay_threshold = entry_qay_threshold
        self.entry_qby_threshold = entry_qby_threshold
        self.entry_qay_prob_threshold = entry_qay_prob_threshold
        self.entry_qby_prob_threshold = entry_qby_prob_threshold

    def store_report(self, report):
        metrics = self.build_metrics().to_frame().T
        to_parquet(
            df=metrics,
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
            "possible_in_debt": self.possible_in_debt,
            "achieved_with_commission": self.achieved_with_commission,
            "max_n_updated": self.max_n_updated,
            "tradable_coins": tuple(self.tradable_coins.tolist()),
            "exit_if_achieved": self.exit_if_achieved,
            "exit_q_threshold": self.exit_q_threshold,
            "entry_qay_threshold": self.entry_qay_threshold,
            "entry_qby_threshold": self.entry_qby_threshold,
            "entry_qay_prob_threshold": self.entry_qay_prob_threshold,
            "entry_qby_prob_threshold": self.entry_qby_prob_threshold,
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

        q = compute_quantile(trade_return, bins=self.bins[position.asset])

        if position.side == "long":
            if q >= self.exit_q_threshold:
                return True

        if position.side == "short":
            if q <= ((self.n_bins - 1) - self.exit_q_threshold):
                return True

        return False

    def run(self, display=True):
        self.build()
        self.initialize()

        for now in tqdm(self.index):
            # Step1: Prepare pricing and signal
            pricing = self.historical_data_dict["pricing"].loc[now]
            qay_prediction = self.historical_data_dict["qay_predictions"].loc[now]
            qby_prediction = self.historical_data_dict["qby_predictions"].loc[now]
            qay_probability = self.historical_data_dict["qay_probabilities"].loc[now]
            qby_probability = self.historical_data_dict["qby_probabilities"].loc[now]

            # Set assets which has signals
            positive_assets = self.tradable_coins[
                (qay_prediction >= self.entry_qay_threshold)
                & (qby_prediction >= self.entry_qby_threshold)
                & (qay_probability >= self.entry_qay_prob_threshold)
                & (qby_probability >= self.entry_qby_prob_threshold)
            ]
            negative_assets = self.tradable_coins[
                (qay_prediction <= (self.n_bins - 1) - self.entry_qay_threshold)
                & (qby_prediction <= (self.n_bins - 1) - self.entry_qby_threshold)
                & (qay_probability >= self.entry_qay_prob_threshold)
                & (qby_probability >= self.entry_qby_prob_threshold)
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
            self.report(value=self.positions, target="historical_positions", now=now)

        report = self.generate_report()
        self.store_report(report=report)

        if display is True:
            display_markdown("#### QAY Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["qay_predictions"],
                labels=self.historical_data_dict["qay_labels"],
            )

            display_markdown("#### QBY Predictions", raw=True)
            self.display_accuracy(
                predictions=self.historical_data_dict["qby_predictions"],
                labels=self.historical_data_dict["qby_labels"],
            )

            self.display_metrics()
            self.display_report(report=report)

        # Remove historical data dict to reduce memory usage
        del self.historical_data_dict
        gc.collect()


if __name__ == "__main__":
    import fire

    fire.Fire(BacktesterV1)
