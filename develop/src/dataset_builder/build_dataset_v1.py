import os
import json
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import combinations
from sklearn import preprocessing
import joblib
from common_utils import make_dirs
from typing import Callable, List, Dict
from pandarallel import pandarallel

pandarallel.initialize()


CONFIG = {
    "rawdata_dir": "../../storage/dataset/rawdata/csv/",
    "data_store_dir": "../../storage/dataset/dataset_60m_v1/",
    "lookahead_window": 60,
    "n_bins": 10,
    "train_ratio": 0.7,
    "scaler_type": "RobustScaler",
}
COLUMNS = ["open", "high", "low", "close"]


def load_rawdata(file_name):
    rawdata = pd.read_csv(file_name, header=0, index_col=0)[COLUMNS]
    rawdata.index = pd.to_datetime(rawdata.index)

    return rawdata


def _build_feature_by_rawdata(rawdata):
    returns_720m = (
        rawdata.pct_change(720, fill_method=None)
        .iloc[1:]
        .rename(columns={key: key + "_return(720)" for key in COLUMNS})
    ).dropna()

    returns_60m = (
        (
            rawdata.pct_change(60, fill_method=None)
            .iloc[1:]
            .rename(columns={key: key + "_return(60)" for key in COLUMNS})
        )
        .dropna()
        .reindex(returns_720m.index)
    )

    returns_1m = (
        (
            rawdata.pct_change(1, fill_method=None)
            .iloc[1:]
            .rename(columns={key: key + "_return(1)" for key in COLUMNS})
        )
        .dropna()
        .reindex(returns_720m.index)
    )

    inner_changes = []
    for column_pair in sorted(list(combinations(COLUMNS, 2))):
        inner_changes.append(
            rawdata[list(column_pair)]
            .pct_change(1, axis=1, fill_method=None)[column_pair[-1]]
            .rename("_".join(column_pair) + "_change")
        )

    inner_changes = pd.concat(inner_changes, axis=1).reindex(returns_720m.index)

    return pd.concat(
        [returns_1m, returns_60m, returns_720m, inner_changes], axis=1
    ).sort_index()


def build_features(file_names):
    features = {}
    for file_name in tqdm(file_names):
        coin_pair = file_name.split("/")[-1].split(".")[0]

        rawdata = load_rawdata(file_name=file_name)
        feature = _build_feature_by_rawdata(rawdata=rawdata)
        features[coin_pair] = feature

    features = pd.concat(features, axis=1).sort_index()

    return features


def build_scaler(features, scaler_type):
    scaler = getattr(preprocessing, scaler_type)()
    scaler.fit(features[features != 0])

    return scaler


def preprocess_features(features, scaler):
    index = features.index
    columns = features.columns

    processed_features = pd.DataFrame(
        scaler.transform(features), index=index, columns=columns
    )

    return processed_features


def compute_quantile(x, bins):
    for idx in range(len(bins) - 1):
        if bins[idx] < x <= bins[idx + 1]:
            return idx

    raise RuntimeError("unreachable")


def _build_bins(rawdata, lookahead_window, n_bins):
    # build fwd_return(window)
    pricing = rawdata["close"].copy().sort_index()
    fwd_return = (
        pricing.pct_change(lookahead_window, fill_method=None)
        .shift(-lookahead_window)
        .rename(f"fwd_return({lookahead_window})")
        .sort_index()
    )

    _, bins = pd.qcut(
        fwd_return[fwd_return != 0].dropna(), n_bins, retbins=True, labels=False
    )
    bins = np.concatenate([[-np.inf], bins[1:-1], [np.inf]])

    return bins


def build_all_bins(file_names, lookahead_window, n_bins):
    all_bins = {}
    for file_name in tqdm(file_names):
        coin_pair = file_name.split("/")[-1].split(".")[0]

        rawdata = load_rawdata(file_name=file_name)
        bins = _build_bins(
            rawdata=rawdata, lookahead_window=lookahead_window, n_bins=n_bins
        )

        all_bins[coin_pair] = bins

    return pd.DataFrame(all_bins)


def _build_qa_label(rawdata, lookahead_window, n_bins):
    # build fwd_return(window)
    pricing = rawdata["close"].copy().sort_index()
    fwd_return = (
        pricing.pct_change(lookahead_window, fill_method=None)
        .shift(-lookahead_window)
        .rename(f"fwd_return({lookahead_window})")
        .sort_index()
    )

    _, bins = pd.qcut(
        fwd_return[fwd_return != 0].dropna(), n_bins, retbins=True, labels=False
    )

    bins = np.concatenate([[-np.inf], bins[1:-1], [np.inf]])

    qa_label = fwd_return.dropna().parallel_apply(partial(compute_quantile, bins=bins))

    return qa_label


def build_qa_labels(file_names, lookahead_window, n_bins):
    qa_labels = []
    for file_name in tqdm(file_names):
        coin_pair = file_name.split("/")[-1].split(".")[0]

        rawdata = load_rawdata(file_name=file_name)
        qa_labels.append(
            _build_qa_label(
                rawdata=rawdata, lookahead_window=lookahead_window, n_bins=n_bins
            ).rename(coin_pair)
        )

    return pd.concat(qa_labels, axis=1).sort_index()


def _build_qb_label(rawdata, lookahead_window, n_bins):
    # build fwd_return(window)
    pricing = rawdata["close"].copy().sort_index()
    fwd_1m_return = pricing.pct_change(1, fill_method=None).shift(-1)
    fwd_return = (
        fwd_1m_return.add(1)
        .rolling(lookahead_window)
        .parallel_apply(lambda x: x.prod())
        .sub(1)
        .shift(-lookahead_window)
    )

    _, bins = pd.qcut(
        fwd_return[fwd_return != 0].dropna(), n_bins, retbins=True, labels=False
    )

    bins = np.concatenate([[-np.inf], bins[1:-1], [np.inf]])

    qb_label = fwd_return.dropna().parallel_apply(partial(compute_quantile, bins=bins))

    return qb_label


def build_qb_labels(file_names, lookahead_window, n_bins):
    qb_labels = []
    for file_name in tqdm(file_names):
        coin_pair = file_name.split("/")[-1].split(".")[0]

        rawdata = load_rawdata(file_name=file_name)
        qb_labels.append(
            _build_qb_label(
                rawdata=rawdata, lookahead_window=lookahead_window, n_bins=n_bins
            ).rename(coin_pair)
        )

    return pd.concat(qb_labels, axis=1).sort_index()


def build_pricing(file_names):
    pricing = []
    for file_name in tqdm(file_names):
        coin_pair = file_name.split("/")[-1].split(".")[0]

        close = load_rawdata(file_name=file_name)["close"].rename(coin_pair)
        pricing.append(close)

    return pd.concat(pricing, axis=1).sort_index()


def store_artifacts(
    features,
    qa_labels,
    qb_labels,
    pricing,
    scaler,
    bins,
    train_ratio,
    params,
    data_store_dir,
):
    # Make dirs
    train_data_store_dir = os.path.join(data_store_dir, "train")
    test_data_store_dir = os.path.join(data_store_dir, "test")
    make_dirs([train_data_store_dir, test_data_store_dir])

    # Store
    boundary_index = int(len(features.index) * train_ratio)
    features.iloc[:boundary_index].to_csv(
        os.path.join(train_data_store_dir, "X.csv"), compression="gzip"
    )
    features.iloc[boundary_index:].to_csv(
        os.path.join(test_data_store_dir, "X.csv"), compression="gzip"
    )

    qa_labels.iloc[:boundary_index].to_csv(
        os.path.join(train_data_store_dir, "QAY.csv"), compression="gzip"
    )
    qa_labels.iloc[boundary_index:].to_csv(
        os.path.join(test_data_store_dir, "QAY.csv"), compression="gzip"
    )

    qb_labels.iloc[:boundary_index].to_csv(
        os.path.join(train_data_store_dir, "QBY.csv"), compression="gzip"
    )
    qb_labels.iloc[boundary_index:].to_csv(
        os.path.join(test_data_store_dir, "QBY.csv"), compression="gzip"
    )

    pricing.iloc[:boundary_index].to_csv(
        os.path.join(train_data_store_dir, "pricing.csv"), compression="gzip"
    )
    pricing.iloc[boundary_index:].to_csv(
        os.path.join(test_data_store_dir, "pricing.csv"), compression="gzip"
    )

    joblib.dump(scaler, os.path.join(data_store_dir, "scaler.pkl"))
    bins.to_csv(os.path.join(data_store_dir, "bins.csv"))

    with open(os.path.join(data_store_dir, "tradable_coins.txt"), "w") as f:
        f.write("\n".join(pricing.columns.tolist()))

    with open(os.path.join(data_store_dir, "params.json"), "w") as f:
        json.dump(params, f)

    print(f"[+] Artifacts are stored")


def build_dataset_v1(
    rawdata_dir=CONFIG["rawdata_dir"],
    data_store_dir=CONFIG["data_store_dir"],
    lookahead_window=CONFIG["lookahead_window"],
    n_bins=CONFIG["n_bins"],
    train_ratio=CONFIG["train_ratio"],
    scaler_type=CONFIG["scaler_type"],
):
    assert scaler_type in ("RobustScaler", "StandardScaler")

    # Make dirs
    make_dirs([data_store_dir])

    # Set file_names
    file_names = sorted(glob(os.path.join(rawdata_dir, "*")))

    # Build features
    features = build_features(file_names)
    scaler = build_scaler(features=features, scaler_type=scaler_type)

    features = preprocess_features(features=features, scaler=scaler)

    # build qa_labels
    qa_labels = build_qa_labels(
        file_names=file_names, lookahead_window=lookahead_window, n_bins=n_bins
    )
    qb_labels = build_qb_labels(
        file_names=file_names, lookahead_window=lookahead_window, n_bins=n_bins
    )

    # Build pricing
    pricing = build_pricing(file_names=file_names)

    # Build bins
    bins = build_all_bins(
        file_names=file_names, lookahead_window=lookahead_window, n_bins=n_bins
    )

    # Masking with common index
    common_index = features.index & qa_labels.index & qb_labels.index
    features = features.reindex(common_index).sort_index()
    qa_labels = qa_labels.reindex(common_index).sort_index()
    qb_labels = qb_labels.reindex(common_index).sort_index()
    pricing = pricing.reindex(common_index).sort_index()

    params = {
        "lookahead_window": lookahead_window,
        "n_bins": n_bins,
        "train_ratio": train_ratio,
        "scaler_type": scaler_type,
    }

    # Store Artifacts
    store_artifacts(
        features=features,
        qa_labels=qa_labels,
        qb_labels=qb_labels,
        pricing=pricing,
        scaler=scaler,
        bins=bins,
        train_ratio=train_ratio,
        params=params,
        data_store_dir=data_store_dir,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(build_dataset_v1)
