import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Ref:
        https://arxiv.org/pdf/1708.02002.pdf

    Notes:
        When the label is unbalanced, FocalLoss performce better than SoftmaxCrossEntropy
        if gamma is 0, it performs same with CrossEntropy
    """

    def __init__(self, weight=None, gamma=1.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )


CRITERIONS = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "ce": nn.CrossEntropyLoss,
    "fl": FocalLoss,
}
