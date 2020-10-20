from itertools import product


def grid(params):
    params = {
        k: list(v) if type(v) == list or type(v) == tuple else [v]
        for k, v in params.items()
    }

    keys = list(params.keys())
    for values in product(*list(params.values())):
        param = dict(zip(keys, values))
        yield param
