from dataclasses import dataclass


@dataclass
class Config:
    @property
    def TARGET_COINS(self):
        return [
            "ADA/USDT",
            "ATOM/USDT",
            "BNB/USDT",
            "BTC/USDT",
            "EOS/USDT",
            "ETH/USDT",
            "LINK/USDT",
            "LTC/USDT",
            "OMG/USDT",
            "THETA/USDT",
            "TRX/USDT",
            "VET/USDT",
            "WAVES/USDT",
            "XLM-USDT",
            "XMR/USDT",
            "XRP/USDT",
            "ZEC/USDT",
        ]


CFG = Config()
