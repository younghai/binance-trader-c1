from collections import OrderedDict

V1_SET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[0.8, 1],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1],
        max_holding_minutes=30,
        entry_ratio=0.09,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        positive_entry_threshold=[9],
        negative_entry_threshold=[9],
        exit_threshold="auto",
        positive_probability_threshold=[5, 6, 7],
        negative_probability_threshold=[9, "9*1.25", "9*1.5"],
        adjust_prediction=False,
        possible_in_debt=False,
    )
)

V1_SET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[0.8, 1],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1],
        max_holding_minutes=30,
        entry_ratio=0.1,
        commission={"entry": 0.0, "exit": 0.0, "spread": 0.0},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        positive_entry_threshold=[8, 9],
        negative_entry_threshold=[8, 9],
        exit_threshold="auto",
        positive_probability_threshold=[5, 6, 7, 8],
        negative_probability_threshold=[8, 9, "9*1.25", "9*1.5"],
        adjust_prediction=False,
    )
)


V1_CSET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[0.8, 1],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1],
        max_holding_minutes=30,
        entry_ratio=0.09,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        positive_entry_threshold=[9],
        negative_entry_threshold=[9],
        exit_threshold="auto",
        positive_probability_threshold=[5, 6, 7],
        negative_probability_threshold=[9, "9*1.25", "9*1.5"],
        adjust_prediction=False,
        possible_in_debt=False,
    )
)

V1_CSET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[0.8, 1],
        achieved_with_commission=True,
        min_holding_minutes=[0, 1],
        max_holding_minutes=30,
        entry_ratio=0.1,
        commission={"entry": 0.0004, "exit": 0.0002, "spread": 0.0004},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        positive_entry_threshold=[8, 9],
        negative_entry_threshold=[8, 9],
        exit_threshold="auto",
        positive_probability_threshold=[5, 6, 7, 8],
        negative_probability_threshold=[8, 9, "9*1.25", "9*1.5"],
        adjust_prediction=False,
    )
)
