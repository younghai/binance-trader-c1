from .backbone_v1 import BackboneV1, identity
import math
import torch.nn as nn
from trainer.modules.block_1d import NORMS
from trainer.modules import acts


class StackBackboneV1(BackboneV1):
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
        self.pred_fc = nn.Linear(in_channels, 4)
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

    def forward(self, x, id):
        B, _, _ = x.size()
        out = self.blocks(self.first_conv(x))
        out = self.global_avg_pool(self.act(self.norm(out))).view(B, -1)

        preds = self.pred_fc(out) + (out * self.embed(id)).sum(axis=-1, keepdim=True)

        return (
            preds[:, 0],
            self.last_sigmoid(preds[:, 1]),
            preds[:, 2],
            self.last_sigmoid(preds[:, 3]),
        )
