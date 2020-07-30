import os
import collections

import torch
from tqdm import tqdm
import pytorch_tools as pt
from pytorch_tools.utils.misc import listify, to_numpy, AverageMeter

from src.utils import EXTRACTOR_FROM_NAME
from src.data import crop_patches


class ConsoleLogger(pt.fit_wrapper.callbacks.ConsoleLogger):
    """Prints training progress to console for monitoring."""

    def __init__(self, metrics=None):
        """
        Args:
            metrics: List of metrics to log. Logs everything by default"""
        self.metrics = metrics

    def on_loader_begin(self):
        if hasattr(tqdm, "_instances"):  # prevents many printing issues
            tqdm._instances.clear()
        stage_str = "train" if self.state.is_train else "validat"
        desc = f"Epoch {self.state.epoch_log:2d}/{self.state.num_epochs}. {stage_str}ing"
        self.pbar = tqdm(total=self.state.epoch_size, desc=desc, ncols=0)

    def on_loader_end(self):
        # update to avg
        desc = collections.OrderedDict({"Loss": f"{self.state.loss_meter.avg:.4f}"})
        for metric in self.state.metric_meters.values():
            if (self.metrics is None) or (metric.name in self.metrics):
                desc.update({metric.name: f"{metric.avg:.3f}"})
        self.pbar.set_postfix(**desc)
        self.pbar.update()
        self.pbar.close()

    def on_batch_end(self):
        desc = collections.OrderedDict({"Loss": f"{self.state.loss_meter.avg_smooth:.4f}"})
        for metric in self.state.metric_meters.values():
            if (self.metrics is None) or (metric.name in self.metrics):
                desc.update({metric.name: f"{metric.avg:.3f}"})
        self.pbar.set_postfix(**desc)
        self.pbar.update()


class CheckpointSaver(pt.fit_wrapper.callbacks.CheckpointSaver):
    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best):
            ep = self.state.epoch_log
            if self.verbose:
                self.state.logger.info(
                    f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}"
                )
            self.best = current
            save_name = os.path.join(self.save_dir, self.save_name.format(monitor=self.monitor))
            self._save_checkpoint(save_name, current)

    def _save_checkpoint(self, path, metric_value):
        if hasattr(self.state.model, "module"):  # used for saving DDP models
            state_dict = self.state.model.module.state_dict()
        else:
            state_dict = self.state.model.state_dict()
        save_dict = {"epoch": self.state.epoch, "state_dict": state_dict, "value": metric_value}
        if self.include_optimizer:
            save_dict["optimizer"] = self.state.optimizer.state_dict()
        torch.save(save_dict, path)

    def on_end(self):
        self.state.logger.info(f"Best {self.monitor}: {self.best:.4f}")


class TensorBoard(pt.fit_wrapper.callbacks.TensorBoard):
    """
    Saves training and validation statistics for TensorBoard
    Args:
        log_dir: path where to store logs
        log_every: how often to write logs during training
        num_images: Number of images to log. Each image is saved in a separate tab!
    """

    def __init__(self, log_dir: str, log_every: int = 20, num_images: int = 3):
        super().__init__(log_dir, log_every=log_every)
        self.num_images = num_images

        # Flag to save first batch for vizualization
        self.saved = False

    def on_batch_end(self):
        super().on_batch_end()

        # Save first validation batch for plotting
        if not self.state.is_train and not self.saved:
            self.input = self.state.input

    def on_epoch_end(self):
        super().on_epoch_end()

        # Save images only on validation epochs
        if not self.state.is_train:
            data, target = self.input
            output = self.state.model(data)

            for i in range(self.num_images):
                # Concat along X axis and clip to [0, 1] range
                final_image = torch.cat([data[i], output[i], target[i]], dim=2)
                self.writer.add_image(f'val_image/{i}', final_image, self.state.epoch)


class FeatureLoaderMetrics(pt.fit_wrapper.callbacks.Callback):
    r"""
    Args:
        metrics: List with pre-inited metric objects
        feature_extractor: Name of model used to extract features
    """
    def __init__(self, metrics, feature_extractor: str) -> None:
        super().__init__()
        self.metrics = listify(metrics)
        self.metric_names = [m.name for m in self.metrics]

        # Define feature extractor
        self.extractor_name = feature_extractor
        self.feature_extractor = EXTRACTOR_FROM_NAME[feature_extractor].cuda()

        self.target_features = None
        self.prediction_features = None

    def on_begin(self):
        for name in self.metric_names:
            self.state.metric_meters[name] = AverageMeter(name=name)

    def on_loader_begin(self):
        self.target_features = []
        self.prediction_features = []

    @torch.no_grad()
    def on_batch_end(self):
        _, target = self.state.input
        prediction = self.state.output

        # Extract patches from inputs to increase number of features
        prediction_patches = crop_patches(prediction, size=96, stride=32)
        target_patches = crop_patches(target, size=96, stride=32)

        dataset = torch.utils.data.TensorDataset(prediction_patches, target_patches)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # Extract features
        for batch in loader:
            pred_patches, trgt_patches = batch
            pred_features = torch.nn.functional.adaptive_avg_pool2d(self.feature_extractor(pred_patches), 1)
            trgt_features = torch.nn.functional.adaptive_avg_pool2d(self.feature_extractor(trgt_patches), 1)

            self.prediction_features.append(pred_features)
            self.target_features.append(trgt_features)

    @torch.no_grad()
    def on_loader_end(self):
        # Reduce collected features
        prediction_features = torch.cat(self.prediction_features).squeeze()
        target_features = torch.cat(self.target_features).squeeze()

        for metric, name in zip(self.metrics, self.metric_names):
            value = to_numpy(metric(prediction_features, target_features))
            self.state.metric_meters[name].update(value)
