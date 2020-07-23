r"""Wrappers for PIQ metrics to correctly work with Trainer"""
import piq
import torch


class BRISQUEWrapper(piq.BRISQUELoss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(prediction)


class ISWrapper(torch.nn.Module):
    def __init__(self, num_splits: int = 5):
        super().__init__()
        self.num_splits = num_splits

    def forward(self, prediction_features: torch.Tensor, target_features: torch.Tensor):
        mean, std = piq.inception_score(features=prediction_features, num_splits=self.num_splits)
        return mean


# Return tensor with features, not list
class InceptionV3Wrapper(piq.feature_extractors.InceptionV3):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return super().forward(inp)[0]


class DummyAverageMeter:
    def __init__(self, value, name):
        self.avg = value
        self.name = name
    


class AverageMeter:
    """Computes and stores the average and current value
        Attributes:
            val - last value
            avg - true average
            avg_smooth - smoothed average"""

    def __init__(self, name="Meter", avg_mom=0.9):
        self.avg_mom = avg_mom
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.avg_smooth = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth = self.avg_smooth * self.avg_mom + val * (1 - self.avg_mom)
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def __call__(self, val):
        return self.update(val)