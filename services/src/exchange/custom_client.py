from dataclasses import dataclass
import ccxt
import pandas as pd
import time
from config import CFG
from datetime import datetime
from common_utils_svc import Position

API_REQUEST_DELAY = 0.1  # sec


@dataclass
class CustomClient:
    binance_cli: ccxt.binance = ccxt.binance(
        {
            "apiKey": CFG.EXCHANGE_API_KEY,
            "secret": CFG.EXCHANGE_SECRET_KEY,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
            "hedgeMode": True,
        }
    )
    test_mode: bool = CFG.TEST_MODE

    def __post_init__(self):
        self.target_coins = CFG.TRADABLE_COINS

        self.__set_test_mode()
        self.__set_dual_position_mode()
        self.__set_leverage()
        self.__set_ammount_constraints()

    def __set_test_mode(self):
        if self.test_mode is True:
            self.binance_cli.set_sandbox_mode(True)

    def __set_dual_position_mode(self):
        try:
            self.binance_cli.fapiPrivatePostPositionSideDual(
                {"dualSidePosition": "true"}
            )
        except ccxt.ExchangeError as f:
            pass
        except:
            raise RuntimeError("[!] Failed to set dual position mode")

    def __set_leverage(self):
        for symbol in self.target_coins:
            leverage = 1

            if self.test_mode is True:
                if symbol in ("XMR/USDT"):
                    leverage = 2

            self.binance_cli.fapiPrivate_post_leverage(
                {"symbol": symbol.replace("/", ""), "leverage": leverage}
            )
            time.sleep(API_REQUEST_DELAY)

    def __set_ammount_constraints(self):
        self.ammount_constraints = (
            pd.DataFrame(self.binance_cli.load_markets())
            .xs("limits")
            .apply(lambda x: x["amount"]["min"])
            .to_dict()
        )

    def revision_symbols(self, symbols):
        if "/" in symbols:
            return symbols

        return [
            symbol.replace(CFG.BASE_CURRENCY, "/" + CFG.BASE_CURRENCY)
            for symbol in symbols
        ]

    def get_tickers(self):
        return pd.DataFrame(self.binance_cli.fetch_tickers())

    def get_last_pricing(self):
        return self.get_tickers().xs("last", axis=0).to_dict()

    def get_balance(self):
        for _ in range(20):
            balance = pd.DataFrame(self.binance_cli.fetch_balance())
            if "USDT" in balance:
                return balance

            time.sleep(0.1)

    def get_last_trade_on(self, symbol):
        orders = self.get_closed_orders(symbol=symbol)
        orders = orders[orders["status"] == "FILLED"]

        trade_on = orders.iloc[0]["time"]
        return (
            pd.Timestamp(datetime.utcfromtimestamp(trade_on / 1000))
            .floor("T")
            .tz_localize("UTC")
        )

    def get_positions(self, balance=None, symbol=None):
        if balance is None:
            balance = self.get_balance()

        positions = pd.DataFrame(balance.xs("positions")["info"])
        positions["symbol"] = self.revision_symbols(positions["symbol"])

        if symbol is not None:
            positions = positions[positions["symbol"] == symbol]

        positions["positionAmt"] = (
            positions["positionAmt"].astype(float).map(lambda x: x if x >= 0 else -x)
        )

        return positions

    def get_position_objects(self, symbol=None, with_entry_at=True):
        posis = self.get_positions(symbol=symbol)
        posis = posis[posis["positionAmt"].astype(float) != 0.0]
        assert posis["symbol"].is_unique

        positions = []
        for posi in posis.to_dict(orient="records"):
            position = Position(
                asset=posi["symbol"],
                side=posi["positionSide"].lower(),
                qty=float(posi["positionAmt"]),
                entry_price=float(posi["entryPrice"]),
                entry_at=self.get_last_trade_on(symbol=posi["symbol"])
                if with_entry_at is True
                else None,
            )
            positions.append(position)

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

        try:
            order = self.binance_cli.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params={"positionSide": position},
            )["info"]
        except ccxt.errors.ExchangeError as e:
            if CFG.TEST_MODE is True:
                return None

            raise ccxt.errors.ExchangeError(e)

        order["symbol"] = self.revision_symbols([order["symbol"]])[-1]
        return order

    def exit_order(self, symbol, order_type, position, amount, price=None):
        position = position.upper()
        assert position in ("LONG", "SHORT")

        if position == "LONG":
            side = "sell"
        if position == "SHORT":
            side = "buy"

        try:
            order = self.binance_cli.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params={"positionSide": position},
            )["info"]
        except ccxt.errors.ExchangeError as e:
            if CFG.TEST_MODE is True:
                return None

            raise ccxt.errors.ExchangeError(e)

        order["symbol"] = self.revision_symbols([order["symbol"]])[-1]
        return order

    def get_orders(self, symbol, limit=50):
        orders = self.binance_cli.fetch_orders(symbol=symbol, limit=limit)
        orders = pd.DataFrame(reversed([order["info"] for order in orders]))

        if len(orders) != 0:
            orders["symbol"] = self.revision_symbols(orders["symbol"])
        return orders

    def get_open_orders(self, symbol):
        orders = self.binance_cli.fetch_open_orders(symbol=symbol)
        orders = pd.DataFrame(reversed([order["info"] for order in orders]))

        if len(orders) != 0:
            orders["symbol"] = self.revision_symbols(orders["symbol"])
        return orders

    def get_closed_orders(self, symbol):
        orders = self.binance_cli.fetch_closed_orders(symbol=symbol)
        orders = pd.DataFrame(reversed([order["info"] for order in orders]))

        if len(orders) != 0:
            orders["symbol"] = self.revision_symbols(orders["symbol"])
        return orders

    def cancel_orders(self, symbol):
        orders = self.get_open_orders(symbol=symbol)

        if len(orders) >= 1:
            for id in orders["orderId"]:
                self.binance_cli.cancel_order(id, symbol=symbol)
                print(f"[!]Cancelled: symbol: {symbol}, id: {id}")
                time.sleep(API_REQUEST_DELAY)
