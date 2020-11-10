import os
from dataclasses import dataclass
from .utils import nan_to_zero
from .basic_backtester import BasicBacktester
from tqdm import tqdm
from IPython.display import display_markdown, display
import gc
import json
from common_utils import to_parquet

from config import CFG
from trainer.models import PredictorV1
from data_collector.usecase import Usecase as DCUsecase


@dataclass
class TraderV1:
    dc_usecase = DCUsecase()

    def __post_init__(self):
        self._set_report_params()
        self._build_model()

    def _set_report_params(self):
        self.base_currency = CFG.REPORT_PARAMS["base_currency"]
        self.position_side = CFG.REPORT_PARAMS["position_side"]
        self.entry_ratio = CFG.REPORT_PARAMS["entry_ratio"]
        self.min_holding_minutes = CFG.REPORT_PARAMS["min_holding_minutes"]
        self.max_holding_minutes = CFG.REPORT_PARAMS["max_holding_minutes"]
        self.compound_interest = CFG.REPORT_PARAMS["compound_interest"]
        self.order_criterion = CFG.REPORT_PARAMS["order_criterion"]
        self.exit_if_achieved = CFG.REPORT_PARAMS["exit_if_achieved"]
        self.achieve_ratio = CFG.REPORT_PARAMS["achieve_ratio"]
        self.achieved_with_commission = CFG.REPORT_PARAMS["achieved_with_commission"]
        self.max_n_updated = CFG.REPORT_PARAMS["max_n_updated"]
        self.exit_q_threshold = CFG.REPORT_PARAMS["exit_q_threshold"]
        self.entry_qay_threshold = CFG.REPORT_PARAMS["entry_qay_threshold"]
        self.entry_qby_threshold = CFG.REPORT_PARAMS["entry_qby_threshold"]
        self.entry_qay_prob_threshold = CFG.REPORT_PARAMS["entry_qay_prob_threshold"]
        self.entry_qby_prob_threshold = CFG.REPORT_PARAMS["entry_qby_prob_threshold"]
        self.sum_probs_above_threshold = CFG.REPORT_PARAMS["sum_probs_above_threshold"]

    def _build_model(self):
        self.model = PredictorV1(
            m_config=CFG.MODEL_PARAMS, device="cpu", mode="predict"
        )

    def run(self, display=True):
        self.build()
        self.initialize()

        for now in tqdm(self.index):
            # Step1: Prepare pricing and signal
            pricing = self.historical_data_dict["pricing"].loc[now]
            qay_prediction = self.historical_data_dict["qay_predictions"].loc[now]
            qby_prediction = self.historical_data_dict["qby_predictions"].loc[now]
            qay_probabilities = self.historical_data_dict["qay_probabilities"].loc[now]
            qby_probabilities = self.historical_data_dict["qby_probabilities"].loc[now]

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
