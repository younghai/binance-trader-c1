from torch import nn


NORMS = {
    "BN": lambda num_channels: nn.BatchNorm1d(num_features=num_channels),
    "GN": lambda num_channels: nn.GroupNorm(num_groups=3, num_channels=num_channels)
    if num_channels % 3 == 0
    else nn.GroupNorm(num_groups=2, num_channels=num_channels),
    "LN": lambda num_channels: nn.GroupNorm(num_groups=1, num_channels=num_channels),
    "IN": lambda num_channels: nn.GroupNorm(
        num_groups=num_channels, num_channels=num_channels
    ),
}


def perform_sn(module, sn=False):
    if sn is False:
        return module
    if sn is True:
        return nn.utils.spectral_norm(module)
