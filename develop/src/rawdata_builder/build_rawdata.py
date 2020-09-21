import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from common_utils import make_dirs


CONFIG = {
    "parquet_rawdata_dir": "../../storage/dataset/rawdata/parquet/",
    "csv_rawdata_store_dir": "../../storage/dataset/rawdata/csv/",
    "candidate_coins_path": "./candidate_coins.txt",
    "query_min_start_dt": "2018-06-01",
}


def load_candidate_coins(candidate_coins_path):
    with open(candidate_coins_path, "r") as f:
        candidate_coins = f.read().splitlines()

    return candidate_coins


def build_rawdata(
    parquet_rawdata_dir=CONFIG["parquet_rawdata_dir"],
    candidate_coins_path=CONFIG["candidate_coins_path"],
    query_min_start_dt=CONFIG["query_min_start_dt"],
    csv_rawdata_store_dir=CONFIG["csv_rawdata_store_dir"],
):
    make_dirs([csv_rawdata_store_dir])
    candidate_coins = load_candidate_coins(candidate_coins_path=candidate_coins_path)

    file_list = glob(os.path.join(parquet_rawdata_dir, "*.parquet"))
    file_list = [
        file
        for file in file_list
        if file.split("/parquet/")[-1].split(".parquet")[0] in candidate_coins
    ]

    n_skipped = 0
    for file in tqdm(file_list):
        df = pd.read_parquet(file)[["open", "high", "low", "close", "volume"]]
        df = df.resample("1T").ffill()

        df = df[query_min_start_dt:]
        assert not df.isnull().any().any()

        csv_name = file.split("rawdata/parquet/")[-1].split(".parquet")[0] + ".csv"

        # skip exists data
        if df.index[0] > pd.Timestamp(query_min_start_dt, freq="T"):
            print(f"skipped to save: {csv_name}")
            n_skipped += 1

            continue

        df.index = df.index.tz_localize("utc")
        df.to_csv(csv_rawdata_store_dir + csv_name)

    print(f"skipped: {n_skipped}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_rawdata)
