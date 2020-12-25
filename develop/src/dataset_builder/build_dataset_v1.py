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


CONFIG = {
    "rawdata_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/cleaned/"),
    "data_store_dir": to_abs_path(__file__, "../../storage/dataset/v001/"),
    "lookahead_window": 30,
    "train_ratio": 0.90,
    "scaler_type": "StandardScaler",
}
COLUMNS = ["open", "high", "low", "close"]
RETURN_COLUMNS = ["open", "high", "low", "close"]


def load_rawdata(file_name):
    rawdata = pd.read_parquet(file_name)[COLUMNS]
    rawdata.index = pd.to_datetime(rawdata.index)

    return rawdata


def _build_feature_by_rawdata(rawdata):
    returns_1440m = (
        rawdata[RETURN_COLUMNS]
        .pct_change(1440, fill_method=None)
        .rename(columns={key: key + "_return(1440)" for key in RETURN_COLUMNS})
    ).dropna()

    returns_1320m = (
        (
            rawdata[RETURN_COLUMNS]
            .pct_change(1320, fill_method=None)
            .rename(columns={key: key + "_return(1320)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    madiv_1320m = (
        (
            rawdata[RETURN_COLUMNS]
            .rolling(1320)
            .mean()
            .rename(columns={key: key + "_madiv(1320)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    returns_600m = (
        (
            rawdata[RETURN_COLUMNS]
            .pct_change(330, fill_method=None)
            .rename(columns={key: key + "_return(330)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1410m.index)
    )

    madiv_600m = (
        (
            rawdata[RETURN_COLUMNS]
            .rolling(600)
            .mean()
            .rename(columns={key: key + "_madiv(600)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    returns_240m = (
        (
            rawdata[RETURN_COLUMNS]
            .pct_change(240, fill_method=None)
            .rename(columns={key: key + "_return(240)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    returns_120m = (
        (
            rawdata[RETURN_COLUMNS]
            .pct_change(120, fill_method=None)
            .rename(columns={key: key + "_return(120)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    returns_1m = (
        (
            rawdata[RETURN_COLUMNS]
            .pct_change(1, fill_method=None)
            .rename(columns={key: key + "_return(1)" for key in RETURN_COLUMNS})
        )
        .dropna()
        .reindex(returns_1440m.index)
    )

    inner_changes = []
    for column_pair in sorted(list(combinations(RETURN_COLUMNS, 2))):
        inner_changes.append(
            rawdata[list(column_pair)]
            .pct_change(1, axis=1, fill_method=None)[column_pair[-1]]
            .rename("_".join(column_pair) + "_change")
        )

    inner_changes = pd.concat(inner_changes, axis=1).reindex(returns_1440m.index)

    return pd.concat(
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


def build_features(file_names):
    features = {}
    for file_name in tqdm(file_names):
        coin_pair = get_filename_by_path(file_name)

        rawdata = load_rawdata(file_name=file_name)
        feature = _build_feature_by_rawdata(rawdata=rawdata)
        features[coin_pair] = feature

    features = pd.concat(features, axis=1).sort_index()

    return features


def build_scaler(data, scaler_type):
    scaler = getattr(preprocessing, scaler_type)()
    scaler.fit(data)

    return scaler


def preprocess_data(data, scaler):
    processed_data = pd.DataFrame(
        scaler.transform(data), index=data.index, columns=data.columns
    )

    return processed_data


def _build_label(rawdata, lookahead_window):
    # build fwd_return(window)
    pricing = rawdata["open"].copy().sort_index()
    fwd_return = (
        pricing.pct_change(lookahead_window, fill_method=None)
        .shift(-lookahead_window - 1)
        .rename(f"fwd_return({lookahead_window})")
        .sort_index()
    )[:-lookahead_window]

    return fwd_return


def build_labels(file_names, lookahead_window):
    labels = []
    for file_name in tqdm(file_names):
        coin_pair = get_filename_by_path(file_name)

        rawdata = load_rawdata(file_name=file_name)
        labels.append(
            _build_label(rawdata=rawdata, lookahead_window=lookahead_window).rename(
                coin_pair
            )
        )

    return pd.concat(labels, axis=1).sort_index()


def build_pricing(file_names):
    pricing = []
    for file_name in tqdm(file_names):
        coin_pair = get_filename_by_path(file_name)

        ohlc = load_rawdata(file_name=file_name)[COLUMNS]
        ohlc.columns = pd.MultiIndex.from_tuples(
            zip([coin_pair for _ in range(len(ohlc.columns))], ohlc.columns)
        )
        pricing.append(ohlc.sort_index())

    return pd.concat(pricing, axis=1).sort_index()


def store_artifacts(
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

    with open(os.path.join(data_store_dir, "tradable_coins.txt"), "w") as f:
        f.write("\n".join(pricing.columns.levels[0].tolist()))

    with open(os.path.join(data_store_dir, "params.json"), "w") as f:
        json.dump(params, f)

    del feature_scaler
    del label_scaler
    del params
    gc.collect()
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


def build_dataset_v1(
    rawdata_dir=CONFIG["rawdata_dir"],
    data_store_dir=CONFIG["data_store_dir"],
    lookahead_window=CONFIG["lookahead_window"],
    train_ratio=CONFIG["train_ratio"],
    scaler_type=CONFIG["scaler_type"],
):
    assert scaler_type in ("RobustScaler", "StandardScaler")
    pandarallel.initialize()

    # Make dirs
    make_dirs([data_store_dir])

    # Set file_names
    file_names = sorted(glob(os.path.join(rawdata_dir, "*")))
    assert len(file_names) != 0

    # Build features
    features = build_features(file_names=file_names)
    feature_scaler = build_scaler(data=features, scaler_type=scaler_type)
    features = preprocess_data(data=features, scaler=feature_scaler)

    # build qa_labels
    labels = build_labels(file_names=file_names, lookahead_window=lookahead_window)
    label_scaler = build_scaler(data=labels, scaler_type=scaler_type)
    labels = preprocess_data(data=labels, scaler=label_scaler)

    # Build pricing
    pricing = build_pricing(file_names=file_names)

    # Reduce memory usage
    gc.collect()

    # Masking with common index
    common_index = (features.index & labels.index).sort_values()
    features = features.reindex(common_index)
    labels = labels.reindex(common_index)
    pricing = pricing.reindex(common_index)

    params = {
        "lookahead_window": lookahead_window,
        "train_ratio": train_ratio,
        "scaler_type": scaler_type,
        "features_columns": features.columns.tolist(),
        "labels_columns": labels.columns.tolist(),
    }

    # Store Artifacts
    store_artifacts(
        features=features,
        labels=labels,
        pricing=pricing,
        feature_scaler=feature_scaler,
        label_scaler=label_scaler,
        train_ratio=train_ratio,
        params=params,
        data_store_dir=data_store_dir,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(build_dataset_v1)
