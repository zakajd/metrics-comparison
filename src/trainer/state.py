r"""Defines state that is used later on in trainer.
References:
    https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/fit_wrapper/state.py
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/runner.py
    https://github.com/catalyst-team/gan/blob/master/catalyst_gan/core/state.py
"""
from typing import List

from pytorch_tools.utils.misc import listify, AverageMeter, TimeMeter

NAMES = ["generator", "discriminator"]


class GANState:
    """
    An object that is used to pass internal state during train/valid/infer.
    This class prohibits creating new attributes after init.

    Args:
        models: Generator and Discriminator models
        optimizers: Generator and Discriminator optimizers
        criterions: Generator and Discrimitor losses
        metrics: Optional metrics to measure generator performance.
            All metrics must have `name` attribute. Defaults to None.
    """

    __isfrozen = False

    def __init__(
        self,
        *,
        models: List = None,
        optimizers: List = None,
        criterions: List = None,
        metrics: List = None,
    ):
        assert len(models) == len(optimizers) == len(criterions) == 2, "Only 2 models are supported for now"

        # Base
        self.metrics = listify(metrics)

        self.models = {
            "generator": models[0],
            "discriminator": models[1]
        }
        self.optimizers = {
            "generator": optimizers[0],
            "discriminator": optimizers[1],
        }
        self.criterions = {
            "generator": criterions[0],
            "discriminator": criterions[1],
        }

        # Data pipeline
        self.input = None
        self.output = {key: None for key in NAMES}

        # Counters
        self.num_epochs = 1
        self.epoch = 0
        self.train_loss = {key: None for key in NAMES}
        self.train_metrics = None

        self.val_loss = {key: None for key in NAMES}
        self.val_metrics = None

        self.is_train = True
        self.epoch_size = None
        self.step = None
        self.batch_size = 0

        # Use average meter for ordinary metrics
        self.metric_meters = [AverageMeter(name=m.name) for m in self.metrics]
        self.loss_meter = {key: AverageMeter("loss") for key in NAMES}

        # For Timer callback
        self.timer = TimeMeter()
        self.__is_frozen = True

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError(f"{self} is a frozen class")
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1
