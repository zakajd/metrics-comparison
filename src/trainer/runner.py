r"""
References:
    https://github.com/catalyst-team/catalyst/blob/master/catalyst/core/runner.py
    https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/fit_wrapper/wrapper.py
"""
from typing import List

import torch
from copy import copy
from pytorch_tools.utils.misc import to_numpy

from src.modules.wrappers import AverageMeter
from src.trainer.state import GANState
from src.trainer.callbacks import Callbacks, ConsoleLogger

NAMES = ["generator", "discriminator"]


class GANTrainer:
    """
    Args:
        models: Generator and Discriminator models
        optimizers: Generator and Discriminator optimizers
        criterions: Generator and Discrimitor losses
        metrics (List): Optional metrics to measure generator performance.
            All metrics must have `name` attribute. Defaults to None.
        callbacks (List): List of Callbacks to use. Defaults to ConsoleLogger().
        gradient_clip_val (float): Gradient clipping value. 0 means no clip. Causes ~5% training slowdown
    """

    def __init__(
        self,
        models: List = None,
        optimizers: List = None,
        criterions: List = None,
        metrics: List = None,
        callbacks=ConsoleLogger(),
        gradient_clip_val: float = 0.0,
    ):

        self.state = GANState(
            models=models, optimizers=optimizers, criterions=criterions, metrics=metrics
        )

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)
        self.gradient_clip_val = gradient_clip_val

    def fit(
        self, train_loader, steps_per_epoch=None, val_loader=None, val_steps=None, epochs=1, start_epoch=0,
    ):
        """
        Args:
            train_loader: DataLoader with defined `len` and `batch_size`
            steps_per_epoch (int): How many steps to count as an epochs. Useful
                when epoch is very long or it's not clearly defined. Defaults to None.
            val_loader: Validation DataLoader with defined `len` and `batch_size` Defaults to None.
            val_steps (int): same as `steps_per_epoch` but for val data. Defaults to None.
            epochs (int): Number of epochs to train for. Defaults to 1.
            start_epoch (int): From which epoch to start. Useful on restarts. Defaults to 0.
        """
        self.state.num_epochs = epochs
        self.state.batch_size = train_loader.batch_size if hasattr(train_loader, "batch_size") else 1
        self.callbacks.on_begin()
        for epoch in range(start_epoch, epochs):
            self.state.is_train = True
            self.state.epoch = epoch
            self.callbacks.on_epoch_begin()
            for key in NAMES:
                self.state.models[key].train()
            self._run_loader(train_loader, steps=steps_per_epoch)
            for key in NAMES:
                self.state.train_loss[key] = copy(self.state.loss_meter[key])
            self.state.train_metrics = [copy(m) for m in self.state.metric_meters]

            if val_loader is not None:
                self.evaluate(val_loader, steps=val_steps)
                for key in NAMES:
                    self.state.val_loss[key] = copy(self.state.loss_meter[key])
                self.state.val_metrics = [copy(m) for m in self.state.metric_meters]
            self.callbacks.on_epoch_end()
        self.callbacks.on_end()

    def evaluate(self, loader, steps=None):
        self.state.is_train = False
        for key in NAMES:
            self.state.models[key].eval()
        self._run_loader(loader, steps=steps)
        return self.state.loss_meter["generator"].avg, [m.avg for m in self.state.metric_meters]

    def _make_step(self):

        data, target = self.state.input

        output = self.state.models["generator"](data)
        fake_output = self.state.models["discriminator"](output)
        real_output = self.state.models["discriminator"](data)

        self.state.output["generator"] = output
        self.state.output["discriminator"] = [fake_output, real_output]

        loss = {
            "generator": self.state.criterions["generator"](output, target, fake_output, real_output),
            "discriminator": self.state.criterions["discriminator"](output, target, fake_output, real_output),
        }
        if self.state.is_train:
            # print("\nOut, Trgt, Fake, Real", output.shape, target.shape, fake_output.shape, real_output.shape)
            # print("\n", output.min(), output.max(), target.min(), target.max())

            # ---------- TRAIN GENERATOR --------------
            self.state.optimizers["generator"].zero_grad()
            loss["generator"] = self.state.criterions["generator"](output, target, fake_output, real_output)
            loss["generator"].backward(retain_graph=True)
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.state.models["generator"].parameters(), self.gradient_clip_val)
            self.state.optimizers["generator"].step()

            # ---------- TRAIN DISCRIMINATOR --------------
            output = self.state.models["generator"](data)
            fake_output = self.state.models["discriminator"](output)
            self.state.optimizers["discriminator"].zero_grad()
            loss["discriminator"] = self.state.criterions["discriminator"](output, target, fake_output, real_output)
            loss["discriminator"].backward()
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.state.models["discriminator"].parameters(), self.gradient_clip_val)
            self.state.optimizers["discriminator"].step()

        # Update metrics and loss
        for key in NAMES:
            self.state.loss_meter[key].update(to_numpy(loss[key]))
        output_denormalized, target_denormalized = output.sigmoid(), target.sigmoid()
        for metric, meter in zip(self.state.metrics, self.state.metric_meters):
            meter.update(to_numpy(metric(output_denormalized, target_denormalized)))

    def _reset_state(self):
        r"""Resets losses, metrics, times, etc."""
        for key in NAMES:
            self.state.loss_meter[key].reset()
        # DummyAverageMeter is added to the end, so just delete last part
        counter = 0
        for i, metric in enumerate(self.state.metric_meters):
            if isinstance(metric, AverageMeter):
                metric.reset()
            else:
                counter += 1
        self.state.metric_meters = self.state.metric_meters[:-counter]
        self.state.timer.reset()

    def _run_loader(self, loader, steps=None):
        self._reset_state()
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
