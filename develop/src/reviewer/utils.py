from itertools import product
from collections import OrderedDict


def grid(params):
    def _handle_value(value):
        return list(value) if type(value) == list or type(value) == tuple else [value]

    params = OrderedDict(
        [(key, _handle_value(value=params[key])) for key in sorted(list(params.keys()))]
    )

    keys = list(params.keys())
    for values in product(*list(params.values())):
        param = dict(zip(keys, values))
        yield param
