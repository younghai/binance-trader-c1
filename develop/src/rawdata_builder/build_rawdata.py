import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from common_utils import make_dirs, load_text


CONFIG = {
    "parquet_rawdata_dir": "../../storage/dataset/rawdata/parquet/",
    "csv_rawdata_store_dir": "../../storage/dataset/rawdata/csv/",
    "candidate_assets_path": "./candidate_assets.txt",
    "query_min_start_dt": "2018-01-01",
}


def build_rawdata(
    parquet_rawdata_dir=CONFIG["parquet_rawdata_dir"],
    candidate_assets_path=CONFIG["candidate_assets_path"],
    query_min_start_dt=CONFIG["query_min_start_dt"],
    csv_rawdata_store_dir=CONFIG["csv_rawdata_store_dir"],
):
    make_dirs([csv_rawdata_store_dir])
    candidate_assets = load_text(path=candidate_assets_path)

    file_list = glob(os.path.join(parquet_rawdata_dir, "*.parquet"))
    file_list = [
        file
        for file in file_list
        if file.split("/parquet/")[-1].split(".parquet")[0] in candidate_assets
    ]

    for file in tqdm(file_list):
        df = pd.read_parquet(file)[["open", "high", "low", "close"]]
        df = df.resample("1T").ffill()

        df = df[query_min_start_dt:]
        assert not df.isnull().any().any()

        csv_name = file.split("rawdata/parquet/")[-1].split(".parquet")[0] + ".csv"
        df.index = df.index.tz_localize("utc")
        df.to_csv(csv_rawdata_store_dir + csv_name)

    print("[+] Built rawdata")


if __name__ == "__main__":
    import fire

    fire.Fire(build_rawdata)
