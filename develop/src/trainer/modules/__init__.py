# flake8: noqa
from .dense import dense
from .norms import NORMS
from .conv1d import conv1d
from .inverted_residual import InvertedRes1d, InvertedResUpsample1d
from .residual import FirstBlockDown1d
from torch.nn import AdaptiveAvgPool1d
from .last_layer import last_norm_act
from .self_attention import SelfAttention1d
