from typing import List
import functools

import piq
import torch
import pytorch_tools as pt
from pytorch_tools.utils.misc import AverageMeter

from src.utils import LOSS_FROM_NAME
from src.features.wrappers import BRISQUEWrapper
from src.features.metrics import METRIC_FROM_NAME


# Modify reset function to delete DummyAverageMeters from metrics list before starting new epoch.
class Runner(pt.fit_wrapper.Runner):
    def _run_loader(self, loader, steps=None):
        self.state.loss_meter.reset()
        self.state.metric_meters = [m for m in self.state.metric_meters if isinstance(m, AverageMeter)]

        for metric in self.state.metric_meters:
            metric.reset()
        self.state.epoch_size = steps or len(loader)  # steps overwrites len
        self.callbacks.on_loader_begin()
        with torch.set_grad_enabled(self.state.is_train):
            for i, batch in enumerate(loader):
                if i == self.state.epoch_size:
                    break
                self.state.step = i
                self.state.input = batch
                self.callbacks.on_batch_begin()
                self._make_step()
                self.callbacks.on_batch_end()
        self.callbacks.on_loader_end()
        return


def criterion_from_list(crit_list):
    """expects something like `l1 0.5 ms-ssim 0.5` to construct loss"""
    losses = [LOSS_FROM_NAME[l] for l in crit_list[::2]]
    losses = [l * float(w) for l, w in zip(losses, crit_list[1::2])]
    return functools.reduce(lambda x, y: x + y, losses)


def metrics_from_list(metrics: List, model_name: str = "vgg16", reduction="none"):
    """Initialize metrics and use single feature extractor for all VGG based metrics
    Args:
        metrics: List of metric names
        model_name: One of {`vgg16`, `vgg19`}
        reduction: Type of reduction to use. One of {'none', 'mean'}
    """
    # model = {
    #     "vgg16": torchvision.models.vgg16(pretrained=True, progress=False).features,
    #     "vgg19": torchvision.models.vgg19(pretrained=True, progress=False).features,
    # }

    # layers_dict = {
    #     "vgg16": piq.perceptual.VGG16_LAYERS,
    #     "vgg19": piq.perceptual.VGG19_LAYERS,
    # }
    result = []
    for name in metrics:
        metric = METRIC_FROM_NAME[name]
        if isinstance(metric, piq.ContentLoss) or isinstance(metric, BRISQUEWrapper):
            metric.reduction = reduction
        else:
            metric = functools.partial(metric, reduction=reduction)

        metric.name = name
        result.append(metric)
    return result
