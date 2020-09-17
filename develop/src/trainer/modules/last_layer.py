import torch
from torch import nn
from ml.torch import init as mlinit
from .norms import NORMS, perform_sn


def identity(x):
    return x


class last_norm_act(nn.Module):
    def __init__(
        self, out_channels, activation="selu", normalization=None, sn=False, dropout=0,
    ):
        super().__init__()

        self.norm = identity
        if normalization is not None:
            self.norm = NORMS[normalization.upper()](num_channels=out_channels)

        self.act = getattr(torch, activation, identity)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(x)))
