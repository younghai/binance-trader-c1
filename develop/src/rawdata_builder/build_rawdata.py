import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from common_utils import (
    make_dirs,
    load_text,
    get_filename_by_path,
    to_parquet,
    get_filename_by_path,
    to_abs_path,
)


CONFIG = {
    "raw_rawdata_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/raw/"),
    "cleaned_rawdata_store_dir": to_abs_path(
        __file__, "../../storage/dataset/rawdata/cleaned/"
    ),
    "candidate_assets_path": to_abs_path(__file__, "./candidate_assets.txt"),
    "query_min_start_dt": "2018-01-01",
    "boundary_dt_must_have_data": "2019-09-01",
}


def build_rawdata(
    raw_rawdata_dir=CONFIG["raw_rawdata_dir"],
    cleaned_rawdata_store_dir=CONFIG["cleaned_rawdata_store_dir"],
    candidate_assets_path=CONFIG["candidate_assets_path"],
    query_min_start_dt=CONFIG["query_min_start_dt"],
    boundary_dt_must_have_data=CONFIG["boundary_dt_must_have_data"],
):
    make_dirs([cleaned_rawdata_store_dir])
    candidate_assets = load_text(path=candidate_assets_path)

    file_list = glob(os.path.join(raw_rawdata_dir, "*.parquet"))
    file_list = [
        file for file in file_list if get_filename_by_path(file) in candidate_assets
    ]
    assert len(file_list) != 0

    count_files = 0
    for file in tqdm(file_list):
        df = pd.read_parquet(file)[["open", "high", "low", "close", "volume"]]
        df = df.resample("1T").ffill()

        df = df[query_min_start_dt:]
        filename = get_filename_by_path(file)
        if df.index[0] > pd.Timestamp(boundary_dt_must_have_data):
            print(f"[!] Skiped: {filename}")
            continue

        assert not df.isnull().any().any()

        store_filename = filename + ".parquet.zstd"
        df.index = df.index.tz_localize("utc")
        to_parquet(df=df, path=os.path.join(cleaned_rawdata_store_dir, store_filename))
        count_files += 1

    print(f"[+] Built rawdata: {count_files}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_rawdata)
