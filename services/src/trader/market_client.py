from dataclasses import dataclass
import ccxt
import pandas as pd
import time
from config import CFG

API_REQUEST_DELAY = 0.1  # sec


@dataclass
class MarketClient:
    binance_cli: ccxt.binance = ccxt.binance(
        {
            "apiKey": CFG.EXCHANGE_API_KEY,
            "secret": CFG.EXCHANGE_SECRET_KEY,
            "enableRateLimit": True,
            "options": {"defaultType": "future",},
            "hedgeMode": True,
        }
    )

    def __post_init__(self):
        self.__set_tradable_coins()
        self.__set_dual_position_mode()
        self.__set_leverage()

    def __set_tradable_coins(self):
        list_coins_on_binance = sorted(self.binance_cli.fetch_tickers().keys())
        self.tradable_coins = sorted(
            [
                tradable_coin
                for tradable_coin in CFG.TRADABLE_COINS
                if tradable_coin in list_coins_on_binance
            ]
        )

    def __set_dual_position_mode(self):
        try:
            self.binance_cli.fapiPrivatePostPositionSideDual(
                {"dualSidePosition": "true"}
            )
        except ccxt.BaseError as f:
            pass
        except:
            raise RuntimeError("[!] Failed to set dual position mode")

    def __set_leverage(self):
        for symbol in self.tradable_coins:
            self.binance_cli.fapiPrivate_post_leverage(
                {"symbol": symbol.replace("/", ""), "leverage": 1,}
            )
            time.sleep(API_REQUEST_DELAY)

    def get_tickers(self):
        return pd.DataFrame(self.binance_cli.fetch_tickers())

    def get_balance(self):
        return pd.DataFrame(self.binance_cli.fetch_balance())

    def get_positions(self, balance=None, symbol=None):
        if balance is None:
            balance = self.get_balance()

        positions = pd.DataFrame(balance.xs("positions")["info"])

        if symbol is not None:
            positions = positions[positions["symbol"] == symbol.replace("/", "")]

        return positions

    def get_available_cache(self, balance=None):
        if balance is None:
            balance = self.get_balance()

        return balance.xs(CFG.BASE_CURRENCY)["free"]

    def get_total_cache(self, balance=None):
        if balance is None:
            balance = self.get_balance()

        return balance.xs(CFG.BASE_CURRENCY)["total"]

    def entry_order(self, symbol, order_type, position, amount, price=None):
        position = position.upper()
        assert position in ("LONG", "SHORT")

        if position == "LONG":
            side = "buy"
        if position == "SHORT":
            side = "sell"

        return self.binance_cli.create_order(
            symbol=symbol.replace("/", ""),
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            params={"positionSide": position},
        )

    def exit_order(self, symbol, order_type, position, amount, price=None):
        position = position.upper()
        assert position in ("LONG", "SHORT")

        if position == "LONG":
            side = "sell"
        if position == "SHORT":
            side = "buy"

        return self.binance_cli.create_order(
            symbol=symbol.replace("/", ""),
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            params={"positionSide": position},
        )

    def get_orders(self, symbol, limit=50):
        orders = self.binance_cli.fetch_orders(
            symbol=symbol.replace("/", ""), limit=limit
        )
        orders = pd.DataFrame(reversed([order["info"] for order in orders]))

        return orders

    def get_open_orders(self, symbol):
        orders = self.binance_cli.fetch_open_orders(symbol=symbol.replace("/", ""))
        orders = pd.DataFrame(reversed([order["info"] for order in orders]))

        return orders

    def cancel_orders(self, symbol):
        orders = self.get_open_orders(symbol=symbol)

        if len(orders) >= 1:
            for id in orders["orderId"]:
                self.binance_cli.cancel_order(id, symbol=symbol.replace("/", ""))
                print(f"[!]Cancelled: symbol: {symbol}, id: {id}")
                time.sleep(API_REQUEST_DELAY)
