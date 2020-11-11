import os
import pandas as pd
from dataclasses import dataclass
from common_utils import load_json
from werkzeug.utils import cached_property


@dataclass
class Config:
    @property
    def ENV(self):
        return os.environ

    @property
    def EXCHANGE_API_KEY(self):
        return self.ENV["EXCHANGE_API_KEY"]

    @property
    def EXCHANGE_SECRET_KEY(self):
        return self.ENV["EXCHANGE_SECRET_KEY"]

    @property
    def TEST_MODE(self):
        return self.ENV["TEST_MODE"]

    @cached_property
    def MODEL_PARAMS(self):
        return load_json(f"/app/dev/experiments/{self.ENV['EXP_NAME']}/params.json")[
            "model_config"
        ]

    @cached_property
    def DATA_PARAMS(self):
        data_params = load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/params.json"
        )["data_config"]
        data_params.pop("checkpoint_dir")
        data_params.pop("generate_output_dir")

        return data_params

    @cached_property
    def REPORT_PARAMS(self):
        return load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/reports/params_{self.ENV['REPORT_PREFIX']}_{self.ENV['REPORT_ID']}_{self.ENV['REPORT_BASE_CURRENCY']}.json"
        )

    @cached_property
    def BINS(self):
        return pd.read_csv(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/bins.csv",
            header=0,
            index_col=0,
        )

    @cached_property
    def TRADABLE_COINS(self):
        return [
            tradable_coin.replace("-", "/")
            for tradable_coin in self.REPORT_PARAMS["tradable_coins"]
        ]

    @property
    def BASE_CURRENCY(self):
        return self.REPORT_PARAMS["base_currency"]


CFG = Config()
