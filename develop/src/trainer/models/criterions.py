import torch.nn as nn


CRITERIONS = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
}
