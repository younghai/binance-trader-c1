import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from .utils import data_loader, display_accuracy, Position, compute_quantile
from .basic_backtester import BasicBacktester
from tqdm import tqdm


CONFIG = {
    "position_side": "long",
    "entry_ratio": 0.1,
    "commission": 0.0015,
    "min_holding_minutes": 2,
    "max_holding_minutes": 10,
    "compound_interest": True,
    "possible_in_debt": False,
    "q_threshold": 7,
    "report_store_dir": "../../storage/report/fwd_10m/v001",
}


class BacktesterV1(BasicBacktester):
    def __init__(
        self,
        base_currency,
        bins_path,
        historical_pricing_path,
        historical_predictions_path,
        report_store_dir=CONFIG["report_store_dir"],
        position_side=CONFIG["position"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
        q_threshold=CONFIG["q_threshold"],
    ):
        super().__init__(
            base_currency=base_currency,
            bins_path=bins_path,
            historical_pricing_path=historical_pricing_path,
            historical_predictions_path=historical_predictions_path,
            report_store_dir=report_store_dir,
            position_side=position_side,
            entry_ratio=entry_ratio,
            commission=commission,
            min_holding_minutes=min_holding_minutes,
            max_holding_minutes=max_holding_minutes,
            compound_interest=compound_interest,
            possible_in_debt=possible_in_debt,
            q_threshold=q_threshold,
        )

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

    def deposit_cache(self, profit):
        self.cache = self.cache + profit

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

    def entry_order(self, asset, side, cache_to_order, now):
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

    def compute_profit(self, position, now):
        current_price = self.historical_pricing.loc[now][position.asset]

        assert position.side in ("long", "short")
        if position.side == "long":
            side_multiply = 1
        if position.side == "short":
            side_multiply = -1

        profit_without_commission = (
            (current_price - position.entry_price) * position.qty * side_multiply
        )
        commission_to_order = (current_price * position.qty) * self.commission

        return profit_without_commission - commission_to_order

    def check_if_achieved(self, position, now):
        current_price = self.historical_pricing.loc[now][position.asset]

        trade_return = (current_price - position.entry_price) / position.entry_price
        q = compute_quantile(trade_return, bins=self.bins[position.asset])

        assert position.side in ("long", "short")
        if position.side == "long":
            if q >= self.q_threshold:
                return True

        if position.side == "short":
            if q <= (9 - self.q_threshold):
                return True

        return False

    def exit_order(self, position, now):
        profit = self.compute_profit(position=position, now=now)
        self.deposit_cache(profit=profit)

    def handle_exit(self, positive_assets, negative_assets, now):
        exited_position_idxes = []
        for position_idx, position in self.positions:
            # Handle min_holding_minutes
            if (now - position.entry_at) <= self.min_holding_minutes:
                continue

            # Handle max_holding_minutes
            if (now - position.entry_at) >= self.max_holding_minutes:
                self.exit_order(position=position, now=now)
                exited_position_idxes.append(position_idx)
                continue

            # Handle exit signal
            if (position.side == "long") and (position.asset in negative_assets):
                self.exit_order(position=position, now=now)
                exited_position_idxes.append(position_idx)
                continue

            if (position.side == "short") and (position.asset in positive_assets):
                self.exit_order(position=position, now=now)
                exited_position_idxes.append(position_idx)
                continue

            # Handle achievement
            if self.check_if_achieved(position=position, now=now) is True:
                self.exit_order(position=position, now=now)
                exited_position_idxes.append(position_idx)
                continue

        # Delete exited positions
        for exited_position_idx in list(reversed(sorted(exited_position_idxes))):
            del self.positions[exited_position_idx]

    def compute_capital(self, now):
        # capital = cache + value of positions
        capital = self.cache

        pricing = self.historical_pricing.loc[now]
        for position in self.positions:
            assert position.side in ("long", "short")
            if position.side == "long":
                side_multiply = 1
            if position.side == "short":
                side_multiply = -1

            current_price = pricing[position.asset]
            capital += (
                (current_price - position.entry_price) * position.qty * side_multiply
            )

        return capital

    def run(self, display=True):
        for now in tqdm(self.index):
            predictions = self.historical_predictions.loc[now]

            positive_assets = self.tradable_coins[predictions == 0]
            negative_assets = self.tradable_coins[predictions == 1]

            # Compute how much use cache
            if self.compound_interest is False:
                cache_to_order = self.entry_ratio
            else:
                cache_to_order = self.cache * self.entry_ratio

            # Entry
            if self.position_side in ("long", "longshort"):
                for order_asset in positive_assets:
                    self.entry_order(
                        asset=order_asset,
                        side="long",
                        cache_to_order=cache_to_order,
                        now=now,
                    )

            if self.position_side in ("short", "longshort"):
                for order_asset in negative_assets:
                    self.entry_order(
                        asset=order_asset,
                        side="short",
                        cache_to_order=cache_to_order,
                        now=now,
                    )

            # Exit
            self.handle_exit(
                positive_assets=positive_assets,
                negative_assets=negative_assets,
                now=now,
            )

            # To report
            self.report(value=self.cache, target="historical_cache", now=now)
            self.report(
                value=self.compute_capital(now=now),
                target="historical_capital",
                now=now,
            )

        report = self.generate_report()
        self.store_report(report=report)

        if display is True:
            self.display_accuracy()
            self.display_metrics()
            self.display_report(report=report)


if __name__ == "__main__":
    import fire

    fire.Fire(BacktesterV1)
