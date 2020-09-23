import os
import numpy as np
import pandas as pd
from abc import abstractmethod
from .utils import data_loader, Position, compute_quantile, nan_to_zero
from .basic_backtester import BasicBacktester
from tqdm import tqdm


CONFIG = {
    "report_prefix": "001",
    "position_side": "long",
    "entry_ratio": 0.05,
    "commission": 0.0015,
    "min_holding_minutes": 2,
    "max_holding_minutes": 10,
    "compound_interest": False,
    "possible_in_debt": True,
}


class BacktesterV1(BasicBacktester):
    def __init__(
        self,
        base_currency,
        dataset_dir,
        exp_dir,
        report_prefix=CONFIG["report_prefix"],
        position_side=CONFIG["position_side"],
        entry_ratio=CONFIG["entry_ratio"],
        commission=CONFIG["commission"],
        min_holding_minutes=CONFIG["min_holding_minutes"],
        max_holding_minutes=CONFIG["max_holding_minutes"],
        compound_interest=CONFIG["compound_interest"],
        possible_in_debt=CONFIG["possible_in_debt"],
    ):
        super().__init__(
            base_currency=base_currency,
            dataset_dir=dataset_dir,
            exp_dir=exp_dir,
            report_prefix=report_prefix,
            position_side=position_side,
            entry_ratio=entry_ratio,
            commission=commission,
            min_holding_minutes=min_holding_minutes,
            max_holding_minutes=max_holding_minutes,
            compound_interest=compound_interest,
            possible_in_debt=possible_in_debt,
        )

    def compute_cost_to_order(self, position):
        cache_to_order = position.entry_price * position.qty
        commission_to_order = cache_to_order * self.commission

        return cache_to_order + commission_to_order

    def check_if_executable_order(self, cost):
        if self.possible_in_debt is True:
            return True

        return bool((self.cache - cost) >= 0)

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

                # Update entry_at and qty
                update_position = Position(
                    asset=exist_position.asset,
                    side=exist_position.side,
                    qty=exist_position.qty + position.qty,
                    entry_price=update_entry_price,
                    entry_at=position.entry_at,
                )

                # Compute cost by only current order
                cost = self.compute_cost_to_order(position=position)
                executable_order = self.check_if_executable_order(cost=cost)

                # Update
                if executable_order is True:
                    self.pay_cache(cost=cost)
                    self.positions[idx] = update_position

                    # updated
                    return True

        return False

    def entry_order(self, asset, side, cache_to_order, pricing, now):
        if cache_to_order == 0:
            return

        entry_price = pricing[asset]
        qty = cache_to_order / entry_price

        position = Position(
            asset=asset,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_at=now,
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

    def compute_profit(self, position, pricing, now):
        current_price = pricing[position.asset]

        if position.side == "long":
            profit_without_commission = current_price * position.qty

        if position.side == "short":
            profit_without_commission = position.entry_price * position.qty
            profit_without_commission += (
                (current_price - position.entry_price) * position.qty * -1
            )

        commission_to_order = (current_price * position.qty) * self.commission

        return profit_without_commission - commission_to_order

    def check_if_achieved(self, position, pricing, now):
        current_price = pricing[position.asset]

        diff_price = current_price - position.entry_price
        if diff_price != 0:
            trade_return = diff_price / position.entry_price
        else:
            trade_return = 0

        q = compute_quantile(trade_return, bins=self.bins[position.asset])

        if position.side == "long":
            if q >= self.q_threshold:
                return True

        if position.side == "short":
            if q <= (9 - self.q_threshold):
                return True

        return False

    def exit_order(self, position, pricing, now):
        profit = self.compute_profit(position=position, pricing=pricing, now=now)
        self.deposit_cache(profit=profit)

    def handle_exit(self, positive_assets, negative_assets, pricing, now):
        for position_idx, position in enumerate(self.positions):
            passed_minutes = (
                pd.Timestamp(now) - pd.Timestamp(position.entry_at)
            ).total_seconds() / 60

            # Handle min_holding_minutes
            if passed_minutes <= self.min_holding_minutes:
                continue

            # Handle max_holding_minutes
            if passed_minutes >= self.max_holding_minutes:
                self.exit_order(position=position, pricing=pricing, now=now)
                self.positions[position_idx].is_exited = True
                continue

            # Handle exit signal
            if (position.side == "long") and (position.asset in negative_assets):
                self.exit_order(position=position, pricing=pricing, now=now)
                self.positions[position_idx].is_exited = True
                continue

            if (position.side == "short") and (position.asset in positive_assets):
                self.exit_order(position=position, pricing=pricing, now=now)
                self.positions[position_idx].is_exited = True
                continue

            # Handle achievement
            if (
                self.check_if_achieved(position=position, pricing=pricing, now=now)
                is True
            ):
                self.exit_order(position=position, pricing=pricing, now=now)
                self.positions[position_idx].is_exited = True
                continue

        # Delete exited positions
        self.positions = [
            position for position in self.positions if position.is_exited is not True
        ]

    def compute_capital(self, pricing, now):
        # capital = cache + value of positions
        capital = self.cache

        for position in self.positions:
            current_price = pricing[position.asset]

            if position.side == "long":
                capital += current_price * position.qty

            if position.side == "short":
                capital += position.entry_price * position.qty
                capital += (current_price - position.entry_price) * position.qty * -1

        return capital

    def run(self, display=True):
        self.initialize()

        for now in tqdm(self.index):
            # Step1: Prepare pricing and signal
            pricing = self.historical_pricing.loc[now]
            predictions = self.historical_predictions.loc[now]

            # Set assets which has signals
            positive_assets = self.tradable_coins[predictions == 0]
            negative_assets = self.tradable_coins[predictions == 1]

            # Compute how much use cache
            if self.compound_interest is False:
                cache_to_order = self.entry_ratio
            else:
                cache_to_order = nan_to_zero(value=(self.cache * self.entry_ratio))

            # Entry order
            if self.position_side in ("long", "longshort"):
                for order_asset in positive_assets:
                    self.entry_order(
                        asset=order_asset,
                        side="long",
                        cache_to_order=cache_to_order,
                        pricing=pricing,
                        now=now,
                    )

            if self.position_side in ("short", "longshort"):
                for order_asset in negative_assets:
                    self.entry_order(
                        asset=order_asset,
                        side="short",
                        cache_to_order=cache_to_order,
                        pricing=pricing,
                        now=now,
                    )

            # Exit
            self.handle_exit(
                positive_assets=positive_assets,
                negative_assets=negative_assets,
                pricing=pricing,
                now=now,
            )

            # To report
            self.report(value=self.cache, target="historical_cache", now=now)
            self.report(
                value=self.compute_capital(pricing=pricing, now=now),
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
