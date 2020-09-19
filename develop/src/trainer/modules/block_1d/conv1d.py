import torch
from torch import nn
from ml.torch import init as mlinit
from .norms import NORMS, perform_sn


def identity(x):
    return x


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        dropout=0,
        kernel_size=3,
        padding=1,
        activation="relu",
        normalization="bn",
        sn=False,
    ):
        super().__init__()

        if normalization is not None:
            bias = False

        self.conv1d = perform_sn(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            sn=sn,
        )

        self.norm = identity
        if normalization is not None:
            self.norm = NORMS[normalization.upper()](num_channels=out_channels)

        self.act = getattr(torch, activation, identity)

        mlinit.glorot_uniform_(self.conv1d.weight)
        self.dropout = nn.Dropout2d(dropout)

        if bias:
            nn.init.constant_(self.conv1d.bias, 0)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv1d(x))))
