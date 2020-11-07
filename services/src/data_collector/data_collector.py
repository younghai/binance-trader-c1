import ccxt
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from typing import List
from config import CFG
from database import database as DB
from database import models
from .usecase import Usecase


@dataclass
class DataCollector:
    usecase = Usecase()
    binance_cli: ccxt.binance = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "future",},}
    )
    target_coins: List[str] = tuple(CFG.TARGET_COINS)

    def __post_init__(self):
        DB.init_db()

        self._set_target_coins()
        self._sync_historical_pricing()

    def _set_target_coins(self):
        list_coins_on_binance = sorted(self.binance_cli.fetch_tickers().keys())
        self.target_coins = sorted(
            [
                target_coin
                for target_coin in self.target_coins
                if target_coin in list_coins_on_binance
            ]
        )

    def _sync_historical_pricing(self):
        inserts_pricings = []
        synced_timestamps = None
        for asset in self.target_coins:
            pricing = self._list_historical_pricing(symbol=asset)

            for pricing_row in pricing.reset_index(drop=False).to_dict(
                orient="records"
            ):
                inserts_pricings.append(
                    {
                        "timestamp": pricing_row["date"],
                        "asset": asset,
                        "open": pricing_row["open"],
                        "high": pricing_row["high"],
                        "low": pricing_row["low"],
                        "close": pricing_row["close"],
                        "volume": pricing_row["volume"],
                    }
                )

            if synced_timestamps is None:
                synced_timestamps = pricing.index
            else:
                synced_timestamps = synced_timestamps & pricing.index

        inserts_syncs = [
            {"timestamp": synced_timestamp}
            for synced_timestamp in synced_timestamps.sort_values().tolist()
        ]

        self.usecase.insert_pricings(inserts=inserts_pricings)
        self.usecase.insert_syncs(inserts=inserts_syncs)

    def _list_historical_pricing(self, symbol, limit=1500):
        assert limit < 2000

        if limit >= 1000:
            pricing = self.binance_cli.fetch_ohlcv(
                symbol=symbol, timeframe="1m", limit=1000
            )
            pricing += self.binance_cli.fetch_ohlcv(
                symbol=symbol,
                timeframe="1m",
                limit=(limit + 1) - 1000,
                since=(pricing[0][0] - (60 * 1000 * 1000)),
            )
        else:
            pricing = self.binance_cli.fetch_ohlcv(
                symbol=symbol, timeframe="1m", limit=limit + 1
            )

        pricing = pd.DataFrame(
            pricing, columns=["date", "open", "high", "low", "close", "volume"]
        ).set_index("date")
        pricing.index = pricing.index.map(
            lambda x: datetime.utcfromtimestamp(x / 1000)
        ).tz_localize("UTC")

        # We drop one value always
        return pricing.sort_index()[:-1]


if __name__ == "__main__":
    import fire

    fire.Fire(DataCollector)
