from torch import nn
from .norms import perform_sn
from trainer.modules import acts


get_activation_func = {
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "mish": acts.Mish,
}


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, activation="relu", sn=False):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            perform_sn(
                nn.Linear(in_channels, in_channels // reduction, bias=False), sn=sn
            ),
            get_activation_func[activation](),
            perform_sn(
                nn.Linear(in_channels // reduction, in_channels, bias=False), sn=sn
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
