from collections import OrderedDict

V1_SET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2, 3],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10, 30],
        max_holding_minutes=60,
        entry_ratio=0.055,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_threshold=[0.001, 0.002, 0.004, 0.005, 0.0075, 0.01],
        adjust_prediction=[False, True],
    )
)

V1_SET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2, 3],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10, 30],
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_threshold=[0.001, 0.002, 0.004, 0.005, 0.0075, 0.01],
        adjust_prediction=[False, True],
    )
)

V1_CSET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2, 3],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10, 30],
        max_holding_minutes=60,
        entry_ratio=0.055,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_threshold=[0.001, 0.002, 0.004, 0.005, 0.0075, 0.01],
        adjust_prediction=[False, True],
    )
)

V1_CSET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2, 3],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10, 30],
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_threshold=[0.001, 0.002, 0.004, 0.005, 0.0075, 0.01],
        adjust_prediction=[False, True],
    )
)
