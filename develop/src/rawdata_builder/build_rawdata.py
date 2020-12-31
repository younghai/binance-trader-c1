import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from common_utils_dev import (
    make_dirs,
    load_text,
    get_filename_by_path,
    to_parquet,
    get_filename_by_path,
    to_abs_path,
)


CONFIG = {
    "raw_spot_rawdata_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/raw/spot/"
    ),
    "raw_future_rawdata_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/raw/future/"
    ),
    "cleaned_rawdata_store_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/cleaned/"
    ),
    "candidate_assets_path": to_abs_path(__file__, "./candidate_assets.txt"),
    "query_min_start_dt": "2018-01-01",
    "boundary_dt_must_have_data": "2019-09-01",
}


def build_rawdata(
    raw_spot_rawdata_dir=CONFIG["raw_spot_rawdata_dir"],
    raw_future_rawdata_dir=CONFIG["raw_future_rawdata_dir"],
    cleaned_rawdata_store_dir=CONFIG["cleaned_rawdata_store_dir"],
    candidate_assets_path=CONFIG["candidate_assets_path"],
    query_min_start_dt=CONFIG["query_min_start_dt"],
    boundary_dt_must_have_data=CONFIG["boundary_dt_must_have_data"],
):
    make_dirs([cleaned_rawdata_store_dir])
    candidate_assets = load_text(path=candidate_assets_path)

    count_files = 0
    for candidate_asset in tqdm(candidate_assets):
        spot_file_path = os.path.join(
            raw_spot_rawdata_dir, f"{candidate_asset}.parquet"
        )
        future_file_path = os.path.join(
            raw_future_rawdata_dir, f"{candidate_asset}.parquet"
        )

        spot_df = pd.read_parquet(spot_file_path)[
            ["open", "high", "low", "close"]
        ].sort_index()
        future_df = pd.read_parquet(future_file_path)[
            ["open", "high", "low", "close"]
        ].sort_index()

        df = pd.concat([spot_df[spot_df.index < future_df.index[0]], future_df])
        df = df.resample("1T").ffill()

        df = df[query_min_start_dt:]
        if df.index[0] > pd.Timestamp(boundary_dt_must_have_data):
            print(f"[!] Skiped: {candidate_asset}")
            continue

        assert not df.isnull().any().any()
        assert len(df.index.unique()) == len(df.index)

        store_filename = candidate_asset + ".parquet.zstd"
        df.index = df.index.tz_localize("utc")
        to_parquet(df=df, path=os.path.join(cleaned_rawdata_store_dir, store_filename))
        count_files += 1

    print(f"[+] Built rawdata: {count_files}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_rawdata)
