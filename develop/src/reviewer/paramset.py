from collections import OrderedDict

V1_SET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10],
        max_holding_minutes=30,
        entry_ratio=0.055,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_threshold=[7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        exit_threshold=["auto", 7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        adjust_prediction=[False, True],
    )
)

V1_SET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10],
        max_holding_minutes=30,
        entry_ratio=0.1,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_threshold=[7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        exit_threshold=["auto", 7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        adjust_prediction=[False, True],
    )
)

V1_CSET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10],
        max_holding_minutes=30,
        entry_ratio=0.055,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_threshold=[7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        exit_threshold=["auto", 7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        adjust_prediction=[False, True],
    )
)

V1_CSET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1, 10],
        max_holding_minutes=30,
        entry_ratio=0.1,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_threshold=[7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        exit_threshold=["auto", 7, 8, 9, "9*1.5", "9*2.0", "9*3.0", "9*4.0"],
        adjust_prediction=[False, True],
    )
)
