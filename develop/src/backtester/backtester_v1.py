import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from .utils import data_loader, Position, compute_quantile, nan_to_zero
from .basic_backtester import BasicBacktester
from tqdm import tqdm


CONFIG = {
    "report_prefix": "001",
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
    "exit_q_threshold": 9,
}


class BacktesterV1(BasicBacktester):
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
        super().__init__(
            base_currency=base_currency,
            dataset_dir=dataset_dir,
            exp_dir=exp_dir,
            report_prefix=report_prefix,
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
        self.initialize()

        for now in tqdm(self.index):
            # Step1: Prepare pricing and signal
            pricing = self.historical_data_dict["pricing"].loc[now]
            predictions = self.historical_data_dict["predictions"].loc[now]

            # Set assets which has signals
            positive_assets = self.tradable_coins[predictions == 0]
            negative_assets = self.tradable_coins[predictions == 1]

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
            self.display_accuracy(
                predictions=self.historical_data_dict["predictions"],
                labels=self.historical_data_dict["labels"],
            )
            self.display_metrics()
            self.display_report(report=report)


if __name__ == "__main__":
    import fire

    fire.Fire(BacktesterV1)
