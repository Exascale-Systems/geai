import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Map [-1, 1] normalized values to [0, 1] before computing soft Dice
        pred = (pred.reshape(pred.size(0), -1) + 1) / 2
        target = (target.reshape(target.size(0), -1) + 1) / 2

        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()
