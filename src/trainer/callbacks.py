import os
import math

import torch
# import logging
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


# class TensorBoardGAN(CallbackGAN, TensorBoard):
#     def __init__(self, log_dir, log_every=20):
#         TensorBoard.__init__(self, log_dir, log_every)
#         # Noise to check progress of generator
#         self.first_input = None

#     def on_batch_end(self):
#         # Save first val batch
#         if not self.state.is_train and self.first_input is None:
#             self.first_input = self.state.input

#         self.current_step += self.state.batch_size
#         if self.state.is_train and (self.current_step % self.log_every == 0):
#             self.writer.add_scalar(
#                 # need proper name
#                 "train_/loss_gen",
#                 self.state.loss_meter.val,
#                 self.current_step,
#             )
#             self.writer.add_scalar(
#                 # need proper name
#                 "train_/loss_disc",
#                 self.state.loss_meter.val,
#                 self.current_step,
#             )
#             for m in self.state.metric_meters:
#                 self.writer.add_scalar(f"train_/{m.name}", m.val, self.current_step)

#     def on_epoch_end(self):
#         # Log scalars
#         self.writer.add_scalar("train/loss_gen", self.state.train_loss.avg, self.current_step)
#         self.writer.add_scalar("train/loss_disc", self.state.train_loss_disc.avg, self.current_step)
#         for m in self.state.train_metrics:
#             self.writer.add_scalar(f"train/{m.name}", m.avg, self.current_step)
#         lr_gen = sorted([pg["lr"] for pg in self.state.optimizer.param_groups])[-1]  # largest lr
#         lr_disc = sorted([pg["lr"] for pg in self.state.optimizer_disc.param_groups])[-1]  # largest lr
#         self.writer.add_scalar("train_/lr_gen", lr_gen, self.current_step)
#         self.writer.add_scalar("train_/lr_disc", lr_disc, self.current_step)
#         self.writer.add_scalar("train/epoch", self.state.epoch, self.current_step)

#         # don't log if no val
#         if self.state.val_loss is None:
#             return

#         # Log scalars
#         self.writer.add_scalar("val/loss_gen", self.state.val_loss.avg, self.current_step)
#         self.writer.add_scalar("val/loss_disc", self.state.val_loss_disc.avg, self.current_step)
#         for m in self.state.val_metrics:
#             self.writer.add_scalar(f"val/{m.name}", m.avg, self.current_step)

#         # Log images
#         N = 16
#         output = self.state.model(self.first_input[0])
#         grid_target = torchvision.utils.make_grid(self.first_input[1][:N], nrow=int(math.sqrt(N)), normalize=True)
#         grid_output = torchvision.utils.make_grid(output[:N], nrow=int(math.sqrt(N)), normalize=True)
#         # Concat along X axis
#         final_image = torch.cat([grid_output, grid_target], dim=2)
#         self.writer.add_image(f'Images', final_image, self.state.epoch)
