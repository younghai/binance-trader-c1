import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.modules.block_1d import DenseBlock, TransitionBlock, NORMS
from trainer.modules import acts


def identity(x):
    return x


class BackboneV1(nn.Module):
    def __init__(
        self,
        in_channels,
        n_assets,
        n_blocks=3,
        n_block_layers=6,
        growth_rate=12,
        dropout=0.0,
        channel_reduction=0.5,
        activation="relu",
        normalization="bn",
        seblock=True,
        sablock=True,
    ):
        super(BackboneV1, self).__init__()
        self.in_channels = in_channels
        self.n_assets = n_assets

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.channel_reduction = channel_reduction

        self.activation = activation
        self.normalization = normalization
        self.seblock = seblock
        self.sablock = sablock

        # Build first_conv
        out_channels = 4 * growth_rate
        self.first_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Build blocks
        in_channels = out_channels

        blocks = []
        for idx in range(n_blocks):

            blocks.append(
                self._build_block(
                    in_channels=in_channels,
                    use_transition_block=True if idx != n_blocks - 1 else False,
                )
            )

            # mutate in_channels for next block
            in_channels = self._compute_out_channels(
                in_channels=in_channels,
                use_transition_block=True if idx != n_blocks - 1 else False,
            )

        self.blocks = nn.Sequential(*blocks)

        # Last layers
        self.norm = identity
        if normalization is not None:
            self.norm = NORMS[normalization.upper()](num_channels=in_channels)

        self.act = getattr(acts, activation)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.embed = nn.Embedding(n_assets, in_channels)
        self.pred_fc = nn.Linear(in_channels, 2)
        self.last_sigmoid = nn.Sigmoid()

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1.0 / n))

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def _compute_out_channels(self, in_channels, use_transition_block=True):
        if use_transition_block is True:
            return int(
                math.floor(
                    (in_channels + (self.n_block_layers * self.growth_rate))
                    * self.channel_reduction
                )
            )

        return in_channels + (self.n_block_layers * self.growth_rate)

    def _build_block(self, in_channels, use_transition_block=True):
        assert use_transition_block in (False, True)

        dense_block = DenseBlock(
            n_layers=self.n_block_layers,
            in_channels=in_channels,
            growth_rate=self.growth_rate,
            dropout=self.dropout,
            activation=self.activation,
            normalization=self.normalization,
            seblock=self.seblock if use_transition_block is True else False,
            sablock=self.sablock if use_transition_block is True else False,
        )

        if use_transition_block is True:
            in_channels = int(in_channels + (self.n_block_layers * self.growth_rate))
            out_channels = int(math.floor(in_channels * self.channel_reduction))
            transition_block = TransitionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=self.dropout,
                activation=self.activation,
                normalization=self.normalization,
            )

            return nn.Sequential(*[dense_block, transition_block])

        return dense_block

    def forward(self, x, id):
        B, _, _ = x.size()
        out = self.blocks(self.first_conv(x))
        out = self.global_avg_pool(self.act(self.norm(out))).view(B, -1)

        preds = self.pred_fc(out) + (out * self.embed(id)).sum(axis=-1, keepdim=True)

        return preds[:, 0], self.last_sigmoid(preds[:, 1])
