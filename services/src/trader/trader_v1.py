import os
import pandas as pd
import numpy as np
from typing import List
from collections import OrderedDict
from dataclasses import dataclass
from .utils import nan_to_zero
from tqdm import tqdm
from IPython.display import display_markdown, display
import gc
import json

from config import CFG
from trainer.models import PredictorV1
from database.usecase import Usecase
from exchange.custom_client import CustomClient
from dataset_builder.build_dataset_v1 import (
    _build_feature_by_rawdata,
    preprocess_features,
)
import joblib


@dataclass
class TraderV1:
    usecase = Usecase()

    def __post_init__(self):
        self.custom_cli = CustomClient()
        self.target_coins = self.custom_cli.target_coins

        self._set_params()
        self._build_model()
        self._build_scaler()

    def _set_params(self):
        # Set params which has dependency on trader logic
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

        self.bins = CFG.BINS
        self.n_bins = CFG.DATASET_PARAMS["n_bins"]

        # Set data builder params
        self.data_builder_params = {}
        self.data_builder_params["scaler_target_feature_columns"] = [
            (column[0].replace("-", "/"), column[1])
            for column in CFG.DATASET_PARAMS["scaler_target_feature_columns"]
        ]
        self.data_builder_params["features_columns"] = [
            (column[0].replace("-", "/"), column[1])
            for column in CFG.DATASET_PARAMS["features_columns"]
        ]
        self.data_builder_params["base_feature_assets"] = [
            base_feature_asset.replace("-", "/")
            for base_feature_asset in CFG.EXP_DATA_PARAMS["base_feature_assets"]
        ]
        self.data_builder_params["winsorize_threshold"] = CFG.EXP_DATA_PARAMS[
            "winsorize_threshold"
        ]
        self.data_builder_params["asset_to_id"] = {
            key.replace("-", "/"): value
            for key, value in CFG.EXP_PARAMS["asset_to_id"].items()
        }
        self.data_builder_params["id_to_asset"] = {
            value: key.replace("-", "/")
            for key, value in CFG.EXP_PARAMS["asset_to_id"].items()
        }

    def _build_model(self):
        self.model = PredictorV1(
            exp_dir=CFG.EXP_DIR,
            m_config=CFG.EXP_MODEL_PARAMS,
            d_config=CFG.EXP_DATA_PARAMS,
            device="cpu",
            mode="predict",
        )

    def _build_scaler(self):
        self.scaler = joblib.load(os.path.join(CFG.EXP_DIR, "scaler.pkl"))

    def _build_features(self, pricing):
        target_coins = sorted(pricing.index.levels[1].unique())

        scaler_target_features = {}
        non_scaler_target_features = {}
        for tradable_coin in target_coins:
            rawdata = pricing.xs(tradable_coin, axis=0, level=1)

            scaler_target_features[tradable_coin] = _build_feature_by_rawdata(
                rawdata=rawdata, scaler_target=True
            )
            non_scaler_target_features[tradable_coin] = _build_feature_by_rawdata(
                rawdata=rawdata, scaler_target=False
            )

        scaler_target_features = pd.concat(scaler_target_features, axis=1).sort_index()[
            self.data_builder_params["scaler_target_feature_columns"]
        ]
        non_scaler_target_features = pd.concat(
            non_scaler_target_features, axis=1
        ).sort_index()

        scaler_target_features = preprocess_features(
            features=scaler_target_features, scaler=self.scaler
        )

        # Concat features
        common_index = scaler_target_features.index & non_scaler_target_features.index
        features = pd.concat(
            [
                scaler_target_features.reindex(common_index),
                non_scaler_target_features.reindex(common_index),
            ],
            axis=1,
            sort=True,
        ).sort_index()[self.data_builder_params["features_columns"]]

        del scaler_target_features
        del non_scaler_target_features
        gc.collect()

        return features

    def _build_inputs(self, features):
        base_features = features[self.data_builder_params["base_feature_assets"]]
        features = features[self.target_coins]

        inputs = []
        for target_coin in self.target_coins:
            to_input = pd.concat([base_features, features[target_coin]], axis=1).astype(
                "float32"
            )

            to_input = np.swapaxes(to_input.values, 0, 1)

            if self.data_builder_params["winsorize_threshold"] is not None:
                to_input = to_input.clip(
                    -self.data_builder_params["winsorize_threshold"],
                    self.data_builder_params["winsorize_threshold"],
                )

            inputs.append(to_input)

        inputs = np.stack(inputs, axis=0)
        ids = [
            self.data_builder_params["asset_to_id"][target_coin]
            for target_coin in self.target_coins
        ]

        return inputs, ids

    def build_prediction_dict(self, last_sync_on):
        query_start_on = last_sync_on - pd.Timedelta(minutes=1469)
        query_end_on = last_sync_on
        pricing = self.usecase.get_pricing(start_on=query_start_on, end_on=query_end_on)

        features = self._build_features(pricing=pricing)
        inputs, ids = self._build_inputs(features=features)

        pred_dict = self.model.predict(
            X=inputs, id=ids, id_to_asset=self.data_builder_params["id_to_asset"]
        )

        return pred_dict

    def build_positive_and_negative_assets(self, pred_dict):
        if self.sum_probs_above_threshold is True:
            positive_qay_probability = pred_dict["qay_probability"][
                pred_dict["qay_probability"].index.get_level_values(1)
                >= self.entry_qay_threshold
            ].sum(axis=0, level=0)
            positive_qby_probability = pred_dict["qby_probability"][
                pred_dict["qby_probability"].index.get_level_values(1)
                >= self.entry_qby_threshold
            ].sum(axis=0, level=0)
            negative_qay_probability = pred_dict["qay_probability"][
                pred_dict["qay_probability"].index.get_level_values(1)
                <= (self.n_bins - 1) - self.entry_qay_threshold
            ].sum(axis=0, level=0)
            negative_qby_probability = pred_dict["qby_probability"][
                pred_dict["qby_probability"].index.get_level_values(1)
                <= (self.n_bins - 1) - self.entry_qby_threshold
            ].sum(axis=0, level=0)
        else:
            positive_qay_probability = pred_dict["qay_probability"].xs(
                self.entry_qay_threshold, axis=0, level=1
            )
            positive_qby_probability = pred_dict["qby_probability"].xs(
                self.entry_qby_threshold, axis=0, level=1
            )
            negative_qay_probability = pred_dict["qay_probability"].xs(
                (self.n_bins - 1) - self.entry_qay_threshold, axis=0, level=1
            )
            negative_qby_probability = pred_dict["qby_probability"].xs(
                (self.n_bins - 1) - self.entry_qby_threshold, axis=0, level=1
            )

        # Set assets which has signals
        positive_assets = self.target_coins[
            (pred_dict["qay_prediction"] >= self.entry_qay_threshold)
            & (pred_dict["qby_prediction"] >= self.entry_qby_threshold)
            & (positive_qay_probability >= self.entry_qay_prob_threshold)
            & (positive_qby_probability >= self.entry_qby_prob_threshold)
        ]
        negative_assets = self.target_coins[
            (
                pred_dict["qay_prediction"]
                <= (self.n_bins - 1) - self.entry_qay_threshold
            )
            & (
                pred_dict["qby_prediction"]
                <= (self.n_bins - 1) - self.entry_qby_threshold
            )
            & (negative_qay_probability >= self.entry_qay_prob_threshold)
            & (negative_qby_probability >= self.entry_qby_prob_threshold)
        ]

        return positive_assets, negative_assets

    def is_executable(self, last_sync_on: pd.Timestamp, now: pd.Timestamp):
        sync_min_delta = int((now - last_sync_on).total_seconds() // 60)

        if sync_min_delta == 1:
            last_trade_on = self.usecase.get_last_trade_on()
            if last_trade_on is None:
                return True
            else:
                if int((now - last_trade_on).total_seconds() // 60) >= 1:
                    return True

        return False

    def run(self):
        # Use timestamp without second info
        now = pd.Timestamp.utcnow().floor("T")
        last_sync_on = self.usecase.get_last_sync_on()

        # if self.is_executable(last_sync_on=last_sync_on, now=now) is True:
        pred_dict = self.build_prediction_dict(last_sync_on=last_sync_on)
        positive_assets, negative_assets = self.build_positive_and_negative_assets(
            pred_dict=pred_dict
        )

        ##### TODO:
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
                    cache_to_order = nan_to_zero(value=(self.cache * self.entry_ratio))
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

        ...
        self.usecase.insert_trade({"timestamp": now})
