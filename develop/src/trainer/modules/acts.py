import torch
import torch.nn as nn
from torch.nn.functional import *


def mish(x):
    return x * torch.tanh(softplus(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return mish(input)
