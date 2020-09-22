import torch.nn as nn
import torch.nn.functional as F
from .norms import NORMS, perform_sn


class FirstBlockDown1d(nn.Module):
    """
    This is for first block with skip connection, without pre-activation structure.
    This is more good way to prevent reducing information
    instead of using convolution on the first layer before bottlneck block.

    1. Selectable activation function
    2. Selectable Normalization method
    3. Selectable Squeeze excitation block

    if downscale is False -> output feature size is same with input
    if downscale is True -> output feature size is output_size = math.ceil(input_size / 2)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation=1,
        activation="relu",
        normalization="bn",
        downscale=False,
        init_channels=None,
        init_normalization=None,
        sn=False,
        bias=False,
    ):
        super().__init__()

        self.normalization = normalization
        self.init_normalization = init_normalization

        stride = 1
        if downscale is True:
            stride = 2

        self.channel_compressor = None
        if (init_channels is not None) and (init_channels != in_channels):
            self.channel_compressor = perform_sn(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=init_channels,
                    kernel_size=1,
                    bias=bias,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

            if init_normalization is not None:
                self.init_n = NORMS[init_normalization.upper()](
                    num_channels=init_channels
                )

            in_channels = init_channels

        self.conv1 = perform_sn(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=bias,
                padding=1,
                stride=stride,
            ),
            sn=sn,
        )
        if normalization is not None:
            self.n1 = NORMS[normalization.upper()](num_channels=out_channels)

        self.conv2 = perform_sn(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                bias=bias,
                padding=1,
                stride=1,
            ),
            sn=sn,
        )

        if downscale is True:
            if in_channels == out_channels:
                self.conv3 = perform_sn(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=bias,
                        padding=1,
                        stride=stride,
                        groups=in_channels,
                    ),
                    sn=sn,
                )

            else:
                self.conv3 = perform_sn(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        bias=bias,
                        padding=1,
                        stride=stride,
                    ),
                    sn=sn,
                )

        elif in_channels != out_channels:
            self.conv3 = perform_sn(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=bias,
                    padding=0,
                    stride=1,
                ),
                sn=sn,
            )

        else:
            self.conv3 = None

        self.act = getattr(F, activation)

    def forward(self, x):
        h = x

        if self.channel_compressor is not None:
            h = self.channel_compressor(h)
            x = self.channel_compressor(x)

            if self.init_normalization is not None:
                h = self.init_n(h)

        h = self.conv1(h)
        if self.normalization is not None:
            h = self.n1(h)
        h = self.act(h)

        h = self.conv2(h)

        if self.conv3:
            x = self.conv3(x)

        h = h + x

        return h
