import torch
import torch.nn as nn
from torch.nn.functional import *


def mish(x):
    return x * torch.tanh(softplus(x))


def tanhexp(x):
    return x * torch.tanh(torch.exp(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return mish(input)


class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()

    def forward(self, input):
        return tanhexp(input)
