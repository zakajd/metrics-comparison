import os
import math

import torch
import logging
from tqdm import tqdm
from pytorch_tools.utils.misc import listify
# from torch.utils.tensorboard import SummaryWriter

from src.trainer import GANState

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

        
    def __init__(self, phases: Dict[List[Dict]], change_every: int = 50):
        self.change_every = change_every
        self.current_lr = {key: None for key in NAMES}
        self.current_mom = {key: None for key in NAMES} 
        
        self.phases = {key: list(map(self.format_phase, phases[key])) for key in NAMES}
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
        new_phase = None
        for phase in reversed(self.phases):
            if self.state.epoch >= phase["ep"][0]:
                new_phase = phase
                break
        if new_phase is None:
            raise Exception("Epoch out of range")
        else:
            self.phase = new_phase

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
        # Update to avg
        desc = OrderedDict()
        for key in NAMES:
            desc.update({f"{key} loss": f"{self.state.loss_meter[key].avg:.3f}"})
            desc.update({m.name: f"{m.avg:.3f}" for m in self.state.metric_meters[key]})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = OrderedDict()
        for key in NAMES:
            desc.update({f"{key} loss": f"{self.state.loss_meter[key].avg_smooth:.3f}"})
            # desc.update({m.name: f"{m.avg_smooth:.3f}" for m in self.state.metric_meters[key]})

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

    def __init__(self, log_dir:str, log_every: int = 40, num_images: int = 2):
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
        self.writer = SummaryWriter(self.log_dir)

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