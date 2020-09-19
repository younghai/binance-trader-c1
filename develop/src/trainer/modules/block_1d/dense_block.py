import torch
import torch.nn as nn
import torch.functional as F
from .norm import NORMS
from .seblock import SEBlock
from .self_attention import SelfAttention1d


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        planes: int = None,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "bn",
        seblock: bool = False,
        sablock: bool = False,
    ):
        super(BottleneckBlock, self).__init__()
        if planes is None:
            planes = out_channels * 4

        # Define blocks
        self.norm1 = NORMS[normalization.upper()](num_channels=in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.norm2 = NORMS[normalization.upper()](num_channels=planes)
        self.conv2 = nn.Conv1d(
            planes, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.act = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)

        # Optional blocks
        self.seblock = None
        if seblock is True:
            self.seblock = SEBlock(in_channels=in_channels, activation=activation)

        self.sablock = None
        if sablock is True:
            self.sablock = SelfAttention1d(in_channels=in_channels)

    def forward(self, x: torch.Tensor):

        after_norm = self.norm1(x)
        if self.seblock is not None:
            after_norm = self.seblock(after_norm)

        after_act = self.act(after_norm)
        if self.sablock is not None:
            after_act = self.sablock(after_act)

        out = self.dropout(self.conv1(after_act))
        out = self.dropout(self.conv2(self.act(self.norm2(out))))

        return torch.cat([x, out], dim=1)


class TransitionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "bn",
    ):
        super(TransitionBlock, self).__init__()
        self.norm = NORMS[normalization.upper()](num_channels=in_channels)
        self.act = getattr(F, activation)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = self.dropout(self.conv(self.act(self.norm(x))))
        return F.avg_pool1d(out, 2)


class DenseBlock(nn.Module):
    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        growth_rate: int,
        planes: int = None,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "bn",
        seblock: bool = False,
        sablock: bool = False,
    ):
        super(DenseBlock, self).__init__()

        layers = [
            BottleneckBlock(
                in_channels=in_channels + (idx * growth_rate),
                out_channels=growth_rate,
                planes=planes,
                dropout=dropout,
                activation=activation,
                normalization=normalization,
                seblock=seblock if idx == (n_layers - 1) else False,
                sablock=sablock if idx == (n_layers - 1) else False,
            )
            for idx in range(n_layers)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
