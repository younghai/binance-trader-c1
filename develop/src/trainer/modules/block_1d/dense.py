import torch
from torch import nn
from .norms import NORMS, perform_sn


def identity(x):
    return x


class Dense(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        bias=True,
        dropout=0,
        activation="relu",
        normalization=None,
        sn=False,
    ):
        super().__init__()

        if normalization is not None:
            bias = False

        self.dense = perform_sn(nn.Linear(input_dim, output_dim, bias=bias), sn=sn)

        self.norm = identity
        if normalization is not None:
            self.norm = NORMS[normalization.upper()](num_channels=output_dim)

        self.act = getattr(torch, activation, identity)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.dense(x))))
