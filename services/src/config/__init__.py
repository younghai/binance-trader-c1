import os
from dataclasses import dataclass
from common_utils import load_json
from werkzeug.utils import cached_property


@dataclass
class Config:
    @property
    def ENV(self):
        return os.environ

    @cached_property
    def MODEL_PARAMS(self):
        return load_json(f"/app/dev/experiments/{self.ENV['EXP_NAME']}/params.json")[
            "model_config"
        ]

    @cached_property
    def REPORT_PARAMS(self):
        return load_json(
            f"/app/dev/experiments/{self.ENV['EXP_NAME']}/reports/params_{self.ENV['REPORT_PREFIX']}_{self.ENV['REPORT_ID']}_{self.ENV['REPORT_BASE_CURRENCY']}.json"
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

    @property
    def EXCHANGE_API_KEY(self):
        return self.ENV["EXCHANGE_API_KEY"]

    @property
    def EXCHANGE_SECRET_KEY(self):
        return self.ENV["EXCHANGE_SECRET_KEY"]


CFG = Config()
