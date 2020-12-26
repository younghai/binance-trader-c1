import os
import gc
import json
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import combinations
from sklearn import preprocessing
import joblib
from common_utils_dev import make_dirs, to_parquet, to_abs_path, get_filename_by_path
from pandarallel import pandarallel
from dataclasses import dataclass


CONFIG = {
    "rawdata_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/cleaned/"),
    "data_store_dir": to_abs_path(__file__, "../../storage/dataset/v001/"),
    "lookahead_window": 30,
    "train_ratio": 0.80,
    "scaler_type": "StandardScaler",
    "winsorize_threshold": 6,
    "query_min_start_dt": "2018-06-01",
}
OHLC = ["open", "high", "low", "close"]


@dataclass
class DatasetBuilder:
    # Defined in running code.
    # Need to give below parameters when build in trader
    tradable_coins = None
    feature_columns = None
    feature_scaler = None
    label_scaler = None

    def build_rawdata(self, file_names, query_min_start_dt):
        def _load_rawdata_row(file_name):
            rawdata = pd.read_parquet(file_name)[OHLC]
            rawdata.index = pd.to_datetime(rawdata.index)
            rawdata = rawdata[query_min_start_dt:]

            return rawdata

        rawdata = {}
        for file_name in tqdm(file_names):
            coin = get_filename_by_path(file_name)
            rawdata[coin] = _load_rawdata_row(file_name=file_name)

        rawdata = pd.concat(rawdata, axis=1).sort_index()

        self.tradable_coins = sorted(rawdata.columns.levels[0].tolist())

        return rawdata[self.tradable_coins]

    def _build_feature_by_rawdata_row(self, rawdata_row):
        returns_1440m = (
            rawdata_row[OHLC]
            .pct_change(1440, fill_method=None)
            .rename(columns={key: key + "_return(1440)" for key in OHLC})
        ).dropna()

        returns_1320m = (
            (
                rawdata_row[OHLC]
                .pct_change(1320, fill_method=None)
                .rename(columns={key: key + "_return(1320)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        madiv_1320m = (
            (
                rawdata_row[OHLC]
                .rolling(1320)
                .mean()
                .rename(columns={key: key + "_madiv(1320)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        returns_600m = (
            (
                rawdata_row[OHLC]
                .pct_change(330, fill_method=None)
                .rename(columns={key: key + "_return(330)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        madiv_600m = (
            (
                rawdata_row[OHLC]
                .rolling(600)
                .mean()
                .rename(columns={key: key + "_madiv(600)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        returns_240m = (
            (
                rawdata_row[OHLC]
                .pct_change(240, fill_method=None)
                .rename(columns={key: key + "_return(240)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        returns_120m = (
            (
                rawdata_row[OHLC]
                .pct_change(120, fill_method=None)
                .rename(columns={key: key + "_return(120)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        returns_1m = (
            (
                rawdata_row[OHLC]
                .pct_change(1, fill_method=None)
                .rename(columns={key: key + "_return(1)" for key in OHLC})
            )
            .dropna()
            .reindex(returns_1440m.index)
        )

        inner_changes = []
        for column_pair in sorted(list(combinations(OHLC, 2))):
            inner_changes.append(
                rawdata_row[list(column_pair)]
                .pct_change(1, axis=1, fill_method=None)[column_pair[-1]]
                .rename("_".join(column_pair) + "_change")
            )

        inner_changes = pd.concat(inner_changes, axis=1).reindex(returns_1440m.index)

        feature = pd.concat(
            [
                returns_1440m,
                returns_1320m,
                madiv_1320m,
                returns_600m,
                madiv_600m,
                returns_240m,
                returns_120m,
                returns_1m,
                inner_changes,
            ],
            axis=1,
        ).sort_index()

        return feature

    def build_features(self, rawdata):
        features = {}
        for coin in tqdm(self.tradable_coins):
            features[coin] = self._build_feature_by_rawdata_row(
                rawdata_row=rawdata[coin]
            )

        features = pd.concat(features, axis=1).sort_index()[self.tradable_coins]

        if self.feature_columns is None:
            self.feature_columns = features.columns
            return features

        return features[self.feature_columns]

    def build_scaler(self, data, scaler_type):
        scaler = getattr(preprocessing, scaler_type)()
        scaler.fit(data)

        return scaler

    def preprocess_features(self, features, winsorize_threshold):
        assert self.feature_scaler is not None

        features = pd.DataFrame(
            self.feature_scaler.transform(features),
            index=features.index,
            columns=features.columns,
        )

        if winsorize_threshold is not None:
            features = (
                features.clip(-winsorize_threshold, winsorize_threshold)
                / winsorize_threshold
            )

        return features

    def preprocess_labels(self, labels, winsorize_threshold):
        assert self.label_scaler is not None

        labels = pd.DataFrame(
            self.label_scaler.transform(labels),
            index=labels.index,
            columns=labels.columns,
        )

        if winsorize_threshold is not None:
            labels = (
                labels.clip(-winsorize_threshold, winsorize_threshold)
                / winsorize_threshold
            )

        return labels

    def _build_label(self, rawdata_row, lookahead_window):
        # build fwd_return(window)
        pricing = rawdata_row["open"].copy().sort_index()
        fwd_return = (
            pricing.pct_change(lookahead_window, fill_method=None)
            .shift(-lookahead_window - 1)
            .rename(f"fwd_return({lookahead_window})")
            .sort_index()
        )[: -lookahead_window - 1]

        return fwd_return

    def build_labels(self, rawdata, lookahead_window):
        labels = []
        for coin in tqdm(self.tradable_coins):
            labels.append(
                self._build_label(
                    rawdata_row=rawdata[coin], lookahead_window=lookahead_window
                ).rename(coin)
            )

        labels = pd.concat(labels, axis=1).sort_index()[self.tradable_coins]

        return labels

    def store_artifacts(
        self,
        features,
        labels,
        pricing,
        feature_scaler,
        label_scaler,
        train_ratio,
        params,
        data_store_dir,
    ):
        # Make dirs
        train_data_store_dir = os.path.join(data_store_dir, "train")
        test_data_store_dir = os.path.join(data_store_dir, "test")
        make_dirs([train_data_store_dir, test_data_store_dir])

        # Store params
        joblib.dump(feature_scaler, os.path.join(data_store_dir, "feature_scaler.pkl"))
        joblib.dump(label_scaler, os.path.join(data_store_dir, "label_scaler.pkl"))

        with open(os.path.join(data_store_dir, "dataset_params.json"), "w") as f:
            json.dump(params, f)

        print(f"[+] Metadata is stored")

        # Store dataset
        boundary_index = int(len(features.index) * train_ratio)

        for file_name, data in [
            ("X.parquet.zstd", features),
            ("Y.parquet.zstd", labels),
            ("pricing.parquet.zstd", pricing),
        ]:
            to_parquet(
                df=data.iloc[:boundary_index],
                path=os.path.join(train_data_store_dir, file_name),
            )

            to_parquet(
                df=data.iloc[boundary_index:],
                path=os.path.join(test_data_store_dir, file_name),
            )

        print(f"[+] Dataset is stored")

    def build(
        self,
        rawdata_dir=CONFIG["rawdata_dir"],
        data_store_dir=CONFIG["data_store_dir"],
        lookahead_window=CONFIG["lookahead_window"],
        train_ratio=CONFIG["train_ratio"],
        scaler_type=CONFIG["scaler_type"],
        winsorize_threshold=CONFIG["winsorize_threshold"],
        query_min_start_dt=CONFIG["query_min_start_dt"],
    ):
        assert scaler_type in ("RobustScaler", "StandardScaler")
        pandarallel.initialize()

        # Make dirs
        make_dirs([data_store_dir])

        # Set file_names
        file_names = sorted(glob(os.path.join(rawdata_dir, "*")))
        assert len(file_names) != 0

        # Build rawdata
        rawdata = self.build_rawdata(
            file_names=file_names, query_min_start_dt=query_min_start_dt
        )
        gc.collect()

        # Build features
        features = self.build_features(rawdata=rawdata)
        self.feature_scaler = self.build_scaler(data=features, scaler_type=scaler_type)
        features = self.preprocess_features(
            features=features, winsorize_threshold=winsorize_threshold
        )
        gc.collect()

        # build labels
        labels = self.build_labels(rawdata=rawdata, lookahead_window=lookahead_window)
        self.label_scaler = self.build_scaler(data=labels, scaler_type=scaler_type)
        labels = self.preprocess_labels(
            labels=labels, winsorize_threshold=winsorize_threshold
        )
        gc.collect()

        # Masking with common index
        common_index = (features.index & labels.index).sort_values()
        features = features.reindex(common_index)
        labels = labels.reindex(common_index)
        pricing = rawdata.reindex(common_index)

        params = {
            "lookahead_window": lookahead_window,
            "train_ratio": train_ratio,
            "scaler_type": scaler_type,
            "features_columns": features.columns.tolist(),
            "labels_columns": labels.columns.tolist(),
            "tradable_coins": self.tradable_coins,
            "winsorize_threshold": winsorize_threshold,
            "query_min_start_dt": query_min_start_dt,
        }

        # Store Artifacts
        self.store_artifacts(
            features=features,
            labels=labels,
            pricing=pricing,
            feature_scaler=self.feature_scaler,
            label_scaler=self.label_scaler,
            train_ratio=train_ratio,
            params=params,
            data_store_dir=data_store_dir,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(DatasetBuilder)
