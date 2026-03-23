import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()
