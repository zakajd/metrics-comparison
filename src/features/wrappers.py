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
