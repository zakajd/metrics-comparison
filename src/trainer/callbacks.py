import os
import math
import collections
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from pytorch_tools.utils.misc import listify, TimeMeter

from src.trainer import GANState
from src.modules.wrappers import DummyAverageMeter
from src.utils import EXTRACTOR_FROM_NAME, METRIC_FROM_NAME
from src.data import crop_patches

NAMES = ["generator", "discriminator"]


class Callback:
    """
    Abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.
    usage example:
    begin
    ---epoch begin (one epoch - one run of every loader)
    ------loader begin
    ---------batch begin
    ---------batch end
    ------loader end
    ---epoch end
    end
    """

    def __init__(self):
        self.state = GANState()

    def set_state(self, state):
        self.state = state

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_begin(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_begin(self):
        pass

    def on_end(self):
        pass


class Callbacks(Callback):
    r"""Combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)

    def set_state(self, state):
        for callback in self.callbacks:
            callback.set_state(state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_begin(self):
        for callback in self.callbacks:
            callback.on_loader_begin()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()


class PhasesScheduler(Callback):
    """
    Scheduler that uses `phases` to process updates.
    Args:
        phases: phases
        change_every: how often to actually change the lr. changing too
            often may slowdown the training
    Example:
        PHASES = {
            "generator":
                [{"ep":[0,8],  "lr":[0,0.1], "mom":0.9, },
                {"ep":[8,24], "lr":[0.1, 0.01], "mode":"cos"},
                {'ep':[24, 30], "lr": 0.001}],
            "discriminator":
                [{"ep":[0,8],  "lr":[0,0.1], "mom":0.9, },
                {"ep":[8,24], "lr":[0.1, 0.01], "mode":"cos"},
                {'ep':[24, 30], "lr": 0.001}],
        }
    """

    def __init__(self, phases: Dict, change_every: int = 50):
        self.change_every = change_every
        self.current_lr = {key: None for key in NAMES}
        self.current_mom = {key: None for key in NAMES}

        self.phases = {key: list(map(self._format_phase, phases[key])) for key in NAMES}
        self.phase = {key: self.phases[key][0] for key in NAMES}
        # Assume that number of epochs for both models are equal
        self.tot_epochs = max([max(p["ep"]) for p in self.phases["generator"]])
        super().__init__()

    def _format_phase(self, phase):
        phase["ep"] = listify(phase["ep"])
        phase["lr"] = listify(phase["lr"])
        phase["mom"] = listify(phase.get("mom", None))  # optional
        if len(phase["lr"]) == 2 or len(phase["mom"]) == 2:
            phase["mode"] = phase.get("mode", "linear")
            assert len(phase["ep"]) == 2, "Linear learning rates must contain end epoch"
        return phase

    @staticmethod
    def _schedule(start, end, pct, mode):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == "linear":
            return start + (end - start) * pct
        elif mode == "cos":
            return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)
        elif mode == "poly":
            gamma = (end / start) ** (1 / 100)
            return start * gamma ** (pct * 100)
        else:
            raise ValueError(f"Mode: `{mode}` is not supported in PhasesScheduler")

    def _get_lr_mom(self, phase, batch_curr):
        batch_tot = self.state.epoch_size
        if len(phase["ep"]) == 1:
            perc = 0
        else:
            ep_start, ep_end = phase["ep"]
            ep_curr, ep_tot = self.state.epoch - ep_start, ep_end - ep_start
            perc = (ep_curr * batch_tot + batch_curr) / (ep_tot * batch_tot)
        if len(phase["lr"]) == 1:
            new_lr = phase["lr"][0]  # constant learning rate
        else:
            lr_start, lr_end = phase["lr"]
            new_lr = self._schedule(lr_start, lr_end, perc, phase["mode"])

        if len(phase["mom"]) == 0:
            new_mom = self.current_mom
        elif len(phase["mom"]) == 1:
            new_mom = phase["mom"][0]
        else:
            mom_start, mom_end = phase["mom"]
            new_mom = self._schedule(mom_start, mom_end, perc, phase["mode"])

        return new_lr, new_mom

    def on_epoch_begin(self):
        new_phase = {}
        for key in NAMES:
            new_phase[key] = None
            for phase in reversed(self.phases[key]):
                if self.state.epoch >= phase["ep"][0]:
                    new_phase[key] = phase
                    break
            if new_phase[key] is None:
                raise Exception("Epoch out of range")
            else:
                self.phase[key] = new_phase[key]

    def on_batch_begin(self):
        for key in NAMES:
            lr, mom = self._get_lr_mom(self.phase[key], self.state.step)
            if (self.current_lr[key] == lr and self.current_mom[key] == mom) or (self.state.step % self.change_every != 0):
                continue

            self.current_lr[key] = lr
            self.current_mom[key] = mom
            for param_group in self.state.optimizers[key].param_groups:
                param_group["lr"] = lr
                param_group["momentum"] = mom


class ConsoleLogger(Callback):
    """Prints training progress to console for monitoring."""

    def on_loader_begin(self):
        if hasattr(tqdm, "_instances"):  # prevents many printing issues
            tqdm._instances.clear()
        stage_str = "train" if self.state.is_train else "validat"
        desc = f"Epoch {self.state.epoch_log:2d}/{self.state.num_epochs}. {stage_str}ing"
        self.pbar = tqdm(total=self.state.epoch_size, desc=desc, ncols=0)

    def on_loader_end(self):
        desc = collections.OrderedDict()
        # Update to avg
        for key in NAMES:
            desc.update({f"{key} loss": f"{self.state.loss_meter[key].avg:.3f}"})

        desc.update({m.name: f"{m.avg:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = collections.OrderedDict()
        for key in NAMES:
            desc.update({f"{key} loss": f"{self.state.loss_meter[key].avg_smooth:.3f}"})

        desc.update({m.name: f"{m.avg_smooth:.3f}" for m in self.state.metric_meters})
        self.pbar.set_postfix(**desc)
        self.pbar.update()


class TensorBoard(Callback):
    """
    Saves training and validation statistics for TensorBoard
    Args:
        log_dir: path where to store logs
        log_every: how often to write logs during training
        num_images: Number of images to log. Each image is saved in a separate tab!
    """

    def __init__(self, log_dir: str, log_every: int = 40, num_images: int = 2):
        super().__init__()
        self.log_dir = log_dir
        self.log_every = log_every
        self.num_images = num_images
        self.writer = None
        self.current_step = 0

        # Flag to save first batch for vizualization
        self.saved = False

    def on_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = torch.utils.tensorboard.SummaryWriter(self.log_dir)

    def on_batch_end(self):
        # Save first validation batch for plotting
        if not self.state.is_train and not self.saved:
            self.input = self.state.input

        self.current_step += self.state.batch_size

        if self.state.is_train and (self.current_step % self.log_every == 0):
            for key in NAMES:
                self.writer.add_scalar(f"train_/{key}_loss", self.state.loss_meter[key].val, self.current_step)
                for m in self.state.metric_meters[key]:
                    self.writer.add_scalar(f"train_/{m.name}", m.val, self.current_step)

    def on_epoch_end(self):
        self.writer.add_scalar("train/epoch", self.state.epoch, self.current_step)
        for key in NAMES:
            self.writer.add_scalar(f"train/{key}_loss", self.state.train_loss[key].avg, self.current_step)
            for m in self.state.train_metrics[key]:
                self.writer.add_scalar(f"train/{m.name}", m.avg, self.current_step)

            lr = sorted([pg["lr"] for pg in self.state.optimizers[key].param_groups])[-1]  # largest lr
            self.writer.add_scalar(f"train_/{key}_lr", lr, self.current_step)

            # Don't log if no val
            if self.state.val_loss[key] is None:
                continue

            self.writer.add_scalar(f"val/{key}_loss", self.state.val_loss[key].avg, self.current_step)
            for m in self.state.val_metrics[key]:
                self.writer.add_scalar(f"val/{m.name}", m.avg, self.current_step)

        # Save images only on validation epochs
        if not self.state.is_train:
            data, target = self.first_input
            output = self.state.model(data)

            for i in range(self.num_images):
                # Concat along X axis
                final_image = torch.cat([data[i], output[i], target[i]], dim=2)
                self.writer.add_image(f'val_image/{i}', final_image, self.state.epoch)

    def on_end(self):
        self.writer.close()


class CheckpointSaver(Callback):
    """
    Save best GENERATOR model every epoch based on loss
    Args:
        save_dir: path to folder where to save the model
        save_name: name of the saved model. can additionally
            add epoch and metric to model save name
        monitor: quantity to monitor. Implicitly prefers validation metrics over train. One of:
            `loss` or name of any metric passed to the runner.
        minimize: Whether to decide to save based on minimizing or maximizing value.
        include_optimizer: if True would also save `optimizers` state_dict.
            This increases checkpoint size 2x times.
        verbose (bool): If `True` reports each time new best is found
    """

    def __init__(
        self,
        save_dir: str,
        save_name="model_{monitor}_{ep}_{metric:.2f}.chpn",
        monitor: str = "loss",
        minimize: bool = True,
        include_optimizer: bool = False,
        verbose=True,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.monitor = monitor

        if minimize:
            self.best = np.inf
            self.monitor_op = np.less
        else:
            self.best = -np.inf
            self.monitor_op = np.greater
        self.include_optimizer = include_optimizer
        self.verbose = verbose

    def on_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best):
            ep = self.state.epoch_log
            if self.verbose:
                logger.info(f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            save_name = os.path.join(self.save_dir, self.save_name.format(ep=ep, metric=current))
            self._save_checkpoint(save_name)

    def _save_checkpoint(self, path):
        save_dict = {"epoch": self.state.epoch}
        for key in NAMES:
            save_dict[key] = self.state.models[key].state_dict()
            if self.include_optimizer:
                save_dict[f"{key}_optimizer"] = self.state.optimizers[key].state_dict()
        torch.save(save_dict, path)

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.state.loss_meter["generator"].avg
        else:
            for metric_meter in self.state.metric_meters:
                if metric_meter.name == self.monitor:
                    value = metric_meter.avg

            for metric in self.state.feature_metric_meters:
                if metric_meter.name == self.monitor:
                    value = metric_meter.avg

        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value


class Timer(Callback):
    """
    Profiles first epoch and prints time spend on data loader and on model.
    Usefull for profiling dataloader code performance
    """

    def __init__(self):
        super().__init__()
        self.has_printed = False
        self.timer = TimeMeter()

    def on_batch_begin(self):
        self.timer.batch_start()

    def on_batch_end(self):
        self.timer.batch_end()

    def on_loader_begin(self):
        self.timer.reset()

    def on_loader_end(self):
        if not self.has_printed:
            self.has_printed = True
            d_time = self.timer.data_time.avg_smooth
            b_time = self.timer.batch_time.avg_smooth
            logger.info(f"\nTimeMeter profiling. Data time: {d_time:.2E}s. Model time: {b_time:.2E}s \n")


class FeatureMetrics(Callback):
    r"""Dirty hack for computation of distribution metrics.
    NOTE: This callback shuold be called before TensorBoard logging, otherwise it won't work properly.

    Args:
        feature_extractor: Name of model used to extract features
        metrics: List with metric names and parameters
    """
    def __init__(self, feature_extractor: str, metrics: List) -> None:
        # Define feature extractor
        self.feature_extractor = EXTRACTOR_FROM_NAME[feature_extractor]

        self.metric_names = metrics[::2]
        self.metrics = [
            METRIC_FROM_NAME[name](**kwargs) for name, kwargs in zip(metrics[::2], metrics[1::2])
        ]
        # Add name for proper logging
        for metric, name in zip(self.metrics, self.metric_names):
            metric.name = f"{name}_{feature_extractor}"

    def on_loader_begin(self):
        r"""Reset state"""
        if self.state.is_train:
            return
        self.prediction_features = []
        self.target_features = []

    def on_batch_end(self):
        _, target = self.state.input
        prediction = self.state.output["generator"]

        # Extract patches from inputs to increase number of features
        prediction_patches = crop_patches(prediction, size=96, stride=32)
        target_patches = crop_patches(target, size=96, stride=32)

        # Extract features from prediction patches
        patch_loader = prediction_patches.view(-1, 10, *prediction_patches.shape[-3:])
        with torch.no_grad():
            for patches in patch_loader:
                features = torch.nn.functional.adaptive_avg_pool2d(self.feature_extractor(patches), 1)
                self.prediction_features.append(features.squeeze())

        # Extract features from target patches
        patch_loader = target_patches.view(-1, 10, *target_patches.shape[-3:])
        with torch.no_grad():
            for patches in patch_loader:
                features = torch.nn.functional.adaptive_avg_pool2d(self.feature_extractor(patches), 1)
                self.target_features.append(features.squeeze())

    def on_loader_end(self):
        # Reduce collected features
        prediction_features = torch.cat(self.prediction_features, dim=0)
        target_features = torch.cat(self.target_features, dim=0)

        for name, metric in zip(self.metric_names, self.metrics):
            score = metric(prediction_features, target_features)
            self.state.metric_meters[name] = DummyAverageMeter(value=score.cpu().numpy())


# class FileLogger(Callback):
#     """Logs loss and metrics every epoch into file.
#     Args:
#         log_dir (str): path where to store the logs
#         logger (logging.Logger): external logger. Default None
#     """

#     def __init__(self, log_dir, logger=None):
#         # logger - already created instance of logger
#         super().__init__()
#         self.logger = logger or self._get_logger(os.path.join(log_dir, "logs.txt"))

#     def on_epoch_begin(self):
#         message = f"Epoch {self.state.epoch_log}"
#         for key in NAMES:
#             message += f"|{key} lr {self.current_lr[key]:.2e}"
#         self.logger.info(message)

#     def on_epoch_end(self):
#         loss, metrics = self.state.train_loss, self.state.train_metrics
#         self.logger.info("Train " + self._format_meters(loss, metrics))
#         if self.state.val_loss is not None:
#             loss, metrics = self.state.val_loss, self.state.val_metrics
#             self.logger.info("Val   " + self._format_meters(loss, metrics))

#     @staticmethod
#     def _get_logger(log_path):
#         logger = logging.getLogger(log_path)
#         logger.setLevel(logging.DEBUG)
#         fh = logging.FileHandler(log_path)
#         fh.setLevel(logging.INFO)
#         formatter = logging.Formatter("[%(asctime)s] %(message)s")
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#         return logger

#     @property
#     def current_lr(self):
#         return sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]

#     @staticmethod
#     def _format_meters(loss, metrics):
#         return f"loss: {loss.avg:.4f} | " + " | ".join(f"{m.name}: {m.avg:.4f}" for m in metrics)
