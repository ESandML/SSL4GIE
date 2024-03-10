import torch.nn as nn


class meanF1Score(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanF1Score, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
            m2 = targets == i
            intersection = m1 * m2

            score += (
                2.0
                * (intersection.sum() + self.smooth)
                / (m1.sum() + m2.sum() + self.smooth)
            )
        return score / self.n_class


class meanPrecision(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanPrecision, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
            m2 = targets == i
            intersection = m1 * m2

            score += (intersection.sum() + self.smooth) / (m1.sum() + self.smooth)
        return score / self.n_class


class meanRecall(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanRecall, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
            m2 = targets == i
            intersection = m1 * m2

            score += (intersection.sum() + self.smooth) / (m2.sum() + self.smooth)
        return score / self.n_class

