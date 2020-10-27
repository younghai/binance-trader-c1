import torch
from torch.nn.functional import *


def mish(x):
    return x * torch.tanh(softplus(x))
