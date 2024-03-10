import torch
import torch.nn as nn


class DiceScore(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        if sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score


class IoU(nn.Module):
    def __init__(self, smooth=1e-8):
        super(IoU, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        if sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (intersection.sum(1) + self.smooth) / (
            m1.sum(1) + m2.sum(1) - intersection.sum(1) + self.smooth
        )
        score = score.sum() / num
        return score


class Precision(nn.Module):
    def __init__(self, smooth=1e-8):
        super(Precision, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        if sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (intersection.sum(1) + self.smooth) / (m1.sum(1) + self.smooth)
        score = score.sum() / num
        return score


class Recall(nn.Module):
    def __init__(self, smooth=1e-8):
        super(Recall, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)

        if sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (intersection.sum(1) + self.smooth) / (m2.sum(1) + self.smooth)
        score = score.sum() / num
        return score

