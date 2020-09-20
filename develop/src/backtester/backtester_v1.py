import numpy as np
from abc import abstractmethod
from .utils import data_loader, display_accuracy, Position
from .basic_backtester import BasicBacktester


CONFIG = {
    "position_side": "long",
    "entry_ratio": 0.1,
    "commission": 0.15,
    "min_holding_minutes": 2,
    "compound_interest": True,
    "possible_in_debt": False,
}


class BacktesterV1(BasicBacktester):
    def __init__(
        self,
        historical_pricing_path,
        historical_predictions_path,
        position_side=CONFIG["position"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
    ):
        assert position_side in ("long", "short", "longshort")
        self.position_side = position_side
        self.commission = commission
        self.min_holding_minutes = min_holding_minutes
        self.compound_interest = compound_interest
        self.possible_in_debt = possible_in_debt

        (
            self.historical_pricing,
            self.historical_predictions,
        ) = self.build_historical_data(
            historical_pricing_path=historical_pricing_path,
            historical_predictions_path=historical_predictions_path,
        )
        self.tradable_coins = self.historical_pricing.columns
        self.index = self.historical_predictions.index

        self.initialize()

    def load_tradable_coins(self, tradable_coins_path):
        with open("tradable_coins_path", "r") as f:
            tradable_coins = f.read().splitlines()

        return tradable_coins

    def build_historical_data(
        self, historical_pricing_path, historical_predictions_path
    ):
        historical_pricing = data_loader(path=historical_pricing_path)
        historical_predictions = data_loader(path=historical_predictions_path)[
            "prediction"
        ]
        historical_predictions.columns = historical_pricing.columns

        return historical_pricing, historical_predictions

    def initialize(self):
        self.positions = []
        self.cache = 1

    def compute_cost_to_order(self, position):
        cache_to_order = position.entry_price * position.qty
        commission_to_order = cache_to_order * self.commission

        return cache_to_order + commission_to_order

    def check_if_executable_order(self, cost):
        if self.possible_in_debt is True:
            return True

        return (self.cache - cost) >= 0

    def pay_cache(self, cost):
        self.cache = self.cache - cost

    def update_position_if_already_have(self, position):
        for idx, exist_position in enumerate(self.positions):
            if (exist_position.asset == position.asset) and (
                exist_position.side == position.side
            ):
                update_entry_price = (
                    (exist_position.entry_price * exist_position.qty)
                    + (position.entry_price * position.qty)
                ) / (exist_position.qty + position.qty)

                # update entry_at and qty
                update_position = Position(
                    asset=exist_position.asset,
                    side=exist_position.side,
                    qty=exist_position.qty + position.qty,
                    entry_price=update_entry_price,
                    entry_at=position.entry_at,
                    base_currency=None,
                )

                cost = self.compute_cost_to_order(position=update_position)
                executable_order = self.check_if_executable_order(cost=cost)

                # Update
                if executable_order is True:
                    self.pay_cache(cost=cost)
                    self.positions[idx] = update_position

                    # updated
                    return True

        return False

    def order(self, asset, side, cache_to_order, now):
        entry_price = self.historical_pricing.loc[now][asset]
        qty = cache_to_order / entry_price

        position = Position(
            asset=asset,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_at=now,
            base_currency=None,
        )

        updated = self.update_position_if_already_have(position=position)
        if updated is True:
            return
        else:
            cost = self.compute_cost_to_order(position=position)
            executable_order = self.check_if_executable_order(cost=cost)

            if executable_order is True:
                self.pay_cache(cost=cost)
                self.positions.append(position)

    def handle_exit(self):
        pass

    def run(self):
        for now in self.index:
            predictions = self.historical_predictions.loc[now]

            positive_assets = self.tradable_coins[predictions == 0]
            negative_assets = self.tradable_coins[predictions == 1]

            # Compute how much use cache
            if self.compound_interest is False:
                cache_to_order = self.entry_ratio
            else:
                cache_to_order = self.cache * self.entry_ratio

            # Order
            if self.position_side in ("long", "longshort"):
                for order_asset in positive_assets:
                    self.order(
                        asset=order_asset,
                        side="long",
                        cache_to_order=cache_to_order,
                        now=now,
                    )

            if self.position_side in ("short", "longshort"):
                for order_asset in negative_assets:
                    self.order(
                        asset=order_asset,
                        side="short",
                        cache_to_order=cache_to_order,
                        now=now,
                    )
