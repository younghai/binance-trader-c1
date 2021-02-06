import os
import kaggle
from common_utils_dev import to_abs_path, make_dirs


SPOT_CONFIG = {
    "dataset_name": "jorijnsmit/binance-full-history",
    "store_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/raw/spot/"),
}

FUTURE_CONFIG = {
    "dataset_name": "nicolaes/binance-futures",
    "store_dir": to_abs_path(__file__, "../../storage/dataset/rawdata/raw/future/"),
}


def download(username, key):
    for config in [SPOT_CONFIG, FUTURE_CONFIG]:
        dataset_name = config["dataset_name"]
        store_dir = config["store_dir"]

        make_dirs([store_dir])

        # Set env to authenticate
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        kaggle.api.authenticate()

        # Download
        kaggle.api.dataset_download_files(dataset_name, path=store_dir, unzip=True)


if __name__ == "__main__":
    import fire

    fire.Fire(download)
