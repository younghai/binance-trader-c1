import os
from .utils import nan_to_zero
from .basic_backtester import BasicBacktester
from tqdm import tqdm
from IPython.display import display_markdown, display
import gc
import json
from common_utils_dev import to_parquet


CONFIG = {
    "report_prefix": "001",
    "detail_report": False,
    "position_side": "longshort",
    "entry_ratio": 0.05,
    "commission": {"entry": 0.0004, "exit": 0.0002, "spread": 0.0002},
    "min_holding_minutes": 1,
    "max_holding_minutes": 60,
    "compound_interest": False,
    "order_criterion": "capital",
    "possible_in_debt": True,
    "exit_if_achieved": True,
    "achieve_ratio": 1,
    "achieved_with_commission": False,
    "max_n_updated": None,
    "entry_qay_threshold": 9,
    "entry_qby_threshold": 9,
    "entry_qay_prob_threshold": 0,
    "entry_qby_prob_threshold": 0,
    "exit_q_threshold": 9,
    "sum_probs_above_threshold": False,
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
        order_criterion=CONFIG["order_criterion"],
        possible_in_debt=CONFIG["possible_in_debt"],
        exit_if_achieved=CONFIG["exit_if_achieved"],
        achieve_ratio=CONFIG["achieve_ratio"],
        achieved_with_commission=CONFIG["achieved_with_commission"],
        max_n_updated=CONFIG["max_n_updated"],
        exit_q_threshold=CONFIG["exit_q_threshold"],
        entry_qay_threshold=CONFIG["entry_qay_threshold"],
        entry_qby_threshold=CONFIG["entry_qby_threshold"],
        entry_qay_prob_threshold=CONFIG["entry_qay_prob_threshold"],
        entry_qby_prob_threshold=CONFIG["entry_qby_prob_threshold"],
        sum_probs_above_threshold=CONFIG["sum_probs_above_threshold"],
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
            order_criterion=order_criterion,
            possible_in_debt=possible_in_debt,
            exit_if_achieved=exit_if_achieved,
            achieve_ratio=achieve_ratio,
            achieved_with_commission=achieved_with_commission,
            max_n_updated=max_n_updated,
            exit_q_threshold=exit_q_threshold,
        )

        self.entry_qay_threshold = entry_qay_threshold
        self.entry_qby_threshold = entry_qby_threshold
        self.entry_qay_prob_threshold = entry_qay_prob_threshold
        self.entry_qby_prob_threshold = entry_qby_prob_threshold
        self.sum_probs_above_threshold = sum_probs_above_threshold

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
            "order_criterion": self.order_criterion,
            "possible_in_debt": self.possible_in_debt,
            "achieved_with_commission": self.achieved_with_commission,
            "max_n_updated": self.max_n_updated,
            "tradable_coins": tuple(self.tradable_coins.tolist()),
            "exit_if_achieved": self.exit_if_achieved,
            "achieve_ratio": self.achieve_ratio,
            "exit_q_threshold": self.exit_q_threshold,
            "entry_qay_threshold": self.entry_qay_threshold,
            "entry_qby_threshold": self.entry_qby_threshold,
            "entry_qay_prob_threshold": self.entry_qay_prob_threshold,
            "entry_qby_prob_threshold": self.entry_qby_prob_threshold,
            "sum_probs_above_threshold": self.sum_probs_above_threshold,
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

    def run(self, display=True):
        self.build()
        self.initialize()

        for idx, now in enumerate(tqdm(self.index)):
            # Step1: Prepare pricing and signal
            pricing = self.historical_data_dict["pricing"].iloc[idx]
            qay_prediction = self.historical_data_dict["qay_predictions"].iloc[idx]
            qby_prediction = self.historical_data_dict["qby_predictions"].iloc[idx]
            qay_probabilities = self.historical_data_dict["qay_probabilities"].iloc[idx]
            qby_probabilities = self.historical_data_dict["qby_probabilities"].iloc[idx]

            if self.sum_probs_above_threshold is True:
                positive_qay_probability = qay_probabilities[
                    qay_probabilities.index.get_level_values(1)
                    >= self.entry_qay_threshold
                ].sum(axis=0, level=0)
                positive_qby_probability = qby_probabilities[
                    qby_probabilities.index.get_level_values(1)
                    >= self.entry_qby_threshold
                ].sum(axis=0, level=0)
                negative_qay_probability = qay_probabilities[
                    qay_probabilities.index.get_level_values(1)
                    <= (self.n_bins - 1) - self.entry_qay_threshold
                ].sum(axis=0, level=0)
                negative_qby_probability = qby_probabilities[
                    qby_probabilities.index.get_level_values(1)
                    <= (self.n_bins - 1) - self.entry_qby_threshold
                ].sum(axis=0, level=0)
            else:
                positive_qay_probability = qay_probabilities.xs(
                    self.entry_qay_threshold, axis=0, level=1
                )
                positive_qby_probability = qby_probabilities.xs(
                    self.entry_qby_threshold, axis=0, level=1
                )
                negative_qay_probability = qay_probabilities.xs(
                    (self.n_bins - 1) - self.entry_qay_threshold, axis=0, level=1
                )
                negative_qby_probability = qby_probabilities.xs(
                    (self.n_bins - 1) - self.entry_qby_threshold, axis=0, level=1
                )

            # Set assets which has signals
            positive_assets = self.tradable_coins[
                (qay_prediction >= self.entry_qay_threshold)
                & (qby_prediction >= self.entry_qby_threshold)
                & (positive_qay_probability >= self.entry_qay_prob_threshold)
                & (positive_qby_probability >= self.entry_qby_prob_threshold)
            ]
            negative_assets = self.tradable_coins[
                (qay_prediction <= (self.n_bins - 1) - self.entry_qay_threshold)
                & (qby_prediction <= (self.n_bins - 1) - self.entry_qby_threshold)
                & (negative_qay_probability >= self.entry_qay_prob_threshold)
                & (negative_qby_probability >= self.entry_qby_prob_threshold)
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
                if self.order_criterion == "cache":
                    if self.cache > 0:
                        cache_to_order = nan_to_zero(
                            value=(self.cache * self.entry_ratio)
                        )
                    else:
                        cache_to_order = 0

                elif self.order_criterion == "capital":
                    # Entry with capital base
                    cache_to_order = nan_to_zero(
                        value=(
                            self.compute_capital(pricing=pricing, now=now)
                            * self.entry_ratio
                        )
                    )

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
