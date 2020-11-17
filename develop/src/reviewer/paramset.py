from collections import OrderedDict

V1_SET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.05,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)

V1_SET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)

V1_SET3 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.001,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=False,
        max_n_updated=None,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)

V1_CSET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.05,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)

V1_CSET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)

V1_CSET3 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.001,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=False,
        max_n_updated=None,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.2, 0.4, 0.5],
        entry_qby_prob_threshold=[0, 0.2, 0.4, 0.5],
        exit_q_threshold=9,
        sum_probs_above_threshold=True,
    )
)


V2_SET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.05,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)

V2_SET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)

V2_SET3 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.001,
        commission={"entry": 0.0, "exit": 0.0},
        compound_interest=False,
        max_n_updated=None,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)

V2_CSET1 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.05,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=True,
        order_criterion="capital",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)

V2_CSET2 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.1,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=True,
        order_criterion="cache",
        max_n_updated=0,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)

V2_CSET3 = OrderedDict(
    dict(
        base_currency="USDT",
        position_side="longshort",
        exit_if_achieved=True,
        achieve_ratio=[1, 2],
        achieved_with_commission=True,
        min_holding_minutes=3,
        max_holding_minutes=60,
        entry_ratio=0.001,
        commission={"entry": 0.0004, "exit": 0.0002},
        compound_interest=False,
        max_n_updated=None,
        entry_qay_threshold=[8, 9],
        entry_qby_threshold=[8, 9],
        entry_qay_prob_threshold=[0, 0.1, 0.2],
        entry_qby_prob_threshold=[0, 0.1, 0.2],
        exit_q_threshold=9,
        sum_probs_above_threshold=False,
    )
)
