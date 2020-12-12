import ccxt
import time
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import List
from config import CFG
from database import database as DB
from database.usecase import Usecase
from logging import getLogger
from common_utils_svc import initialize_main_logger


logger = getLogger("data_collector")
initialize_main_logger()


@dataclass
class DataCollector:
    usecase = Usecase()
    binance_cli: ccxt.binance = ccxt.binance(
        {"timeout": 30000, "options": {"defaultType": "future"},}
    )

    def __post_init__(self):
        DB.init()

        self._set_target_coins()
        self._sync_historical_pricing(now=pd.Timestamp.utcnow())

    def _set_target_coins(self):
        list_coins_on_binance = sorted(self.binance_cli.fetch_tickers().keys())
        self.target_coins = sorted(
            [
                target_coin
                for target_coin in tuple(CFG.TRADABLE_COINS)
                if target_coin in list_coins_on_binance
            ]
        )

    def _build_inserts_dict_to_sync(self, now: pd.Timestamp, limit: int):
        inserts_pricings = []
        synced_timestamps = None
        for asset in self.target_coins:
            pricing = self._list_historical_pricing(now=now, symbol=asset, limit=limit)

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

        return (inserts_pricings, inserts_syncs)

    def _sync_historical_pricing(self, now: pd.Timestamp, limit: int = 1500):
        inserts_pricings, inserts_syncs = self._build_inserts_dict_to_sync(
            now=now, limit=limit
        )

        self.usecase.insert_pricings(inserts=inserts_pricings)
        self.usecase.insert_syncs(inserts=inserts_syncs)
        logger.info(f"[+] Synced: historical pricings")

    def _sync_live_pricing(self, now: pd.Timestamp, limit: int = 10):
        inserts_pricings, inserts_syncs = self._build_inserts_dict_to_sync(
            now=now, limit=limit
        )

        self.usecase.update_pricings(updates=inserts_pricings)
        self.usecase.update_syncs(updates=inserts_syncs)

        self.usecase.delete_old_records(
            table="pricings", limit=1500 * len(self.target_coins)
        )
        self.usecase.delete_old_records(table="syncs", limit=1500)
        self.usecase.delete_old_records(table="trades", limit=1500)

    def _list_historical_pricing(
        self, now: pd.Timestamp, symbol: str, limit: int = 1500
    ):
        assert limit < 2000

        if limit >= 1000:
            pricing = self.binance_cli.fetch_ohlcv(
                symbol=symbol, timeframe="1m", limit=1000
            )

            ext_limit = (limit + 1) - 1000
            pricing += self.binance_cli.fetch_ohlcv(
                symbol=symbol,
                timeframe="1m",
                limit=ext_limit,
                since=(pricing[0][0] - (60 * ext_limit * 1000)),
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
        pricing = pricing.sort_index()
        return pricing[pricing.index < now.floor("T")]

    def _get_minutes_to_sync(self, now: pd.Timestamp):
        last_sync_on = self.usecase.get_last_sync_on()
        minutes_delta = int((now.floor("T") - last_sync_on).total_seconds() // 60)

        return minutes_delta - 1

    def run(self):
        """Definitioin of demon to live sync
        """
        logger.info("[O] Start: demon of data_collector")

        error_count = 0
        while True:
            try:
                now = pd.Timestamp.utcnow()
                minutes_to_sync = self._get_minutes_to_sync(now=now)

                if minutes_to_sync != 0:
                    minutes_to_sync = min(max(minutes_to_sync, 5), 1500)

                    self._sync_live_pricing(now=now, limit=minutes_to_sync)
                    logger.info(f'[+] Synced: {now.floor("T")}')

                    error_count = 0

            except Exception as e:
                error_count += 1

                # Raise error if reached limitation
                if error_count >= 20:
                    # Sleep for Api limitation
                    time.sleep(60)

                    logger.error("[!] Error: ", exc_info=True)
                    raise Exception

                logger.info(f"[!] Synced Failed")
                time.sleep(1)

            time.sleep(0.2)


if __name__ == "__main__":
    import fire

    fire.Fire(DataCollector)
