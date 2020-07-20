# import os
import math
from collections import OrderedDict

import torch
import torchvision
import pytorch_lightning as pl

from src.utils import METRIC_SCALE_FROM_NAME


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch

        # Save for logging
        prediction = self(input)
        loss = self.criterion(prediction, target)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'loss': loss
        })

        return output

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Do not validate Set5 dataset. Use it only for visualization
        if (dataloader_idx == 1) and (self.saved_batch is None):
            self.saved_batch = batch
            return None

        input, target = batch
        prediction = self(input)

        with torch.no_grad():
            loss_val = self.criterion(prediction, target)
            output = OrderedDict({'loss': loss_val})

            # Compute metrics
            for i, name in enumerate(self.metric_names):
                output[name] = self.metrics[i](prediction, target)

            input_features = self.feature_extractor(input)[0].squeeze()

            if self.validation_features is None:
                target_features = self.feature_extractor(target)[0].squeeze()
            else:
                target_features = None

        output['input_features'] = input_features
        output['target_features'] = target_features
        return output

    # def validation_epoch_end(self, outputs):
    def validation_end(self, outputs):
        # Only first validation loader returns metric values
        outputs = outputs[0]

        tqdm_dict = {}
        log_dict = {}

        # Same as log_dict but showed all at one plot
        model_dict = {}

        # Reduce per-image metrics
        self.metric_names.append("loss")
        for metric_name in self.metric_names:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]
                metric_total += metric_value

            # # Not the smartest way to do this
            # log_dict["validation/" + metric_name] = metric_total / len(outputs)
            # model_dict[metric_name] = log_dict["validation/" + metric_name] * METRIC_SCALE_FROM_NAME[metric_name]

        self.metric_names.pop()  # Remove `loss` from this list

        # Collect computed image features into a vector of size (val_size, feat_size)
        all_input_features = [out["input_features"] for out in outputs]
        input_features = torch.cat(all_input_features, dim=0)

        if self.validation_features is None:
            all_target_features = [out["target_features"] for out in outputs]
            self.validation_features = torch.cat(all_target_features, dim=0)

        # Compute feature metrics
        for i, name in enumerate(self.feat_metric_names):
            score = self.feat_metrics[i](input_features.cpu(), self.validation_features.cpu())
            log_dict["validation/" + name] = torch.tensor(score)
            model_dict[metric_name] = log_dict["validation/" + metric_name] * METRIC_SCALE_FROM_NAME[metric_name]

        if "msid" in self.feat_metric_names:
            score = []
            for _ in range(self.hparams.compute_metrics_repeat - 1):
                score.append(
                    self.feat_metrics[self.feat_metric_names.index("msid")](
                        input_features.cpu(),
                        self.validation_features.cpu()
                    )
                )
            log_dict["validation/msid"] = torch.mean(torch.tensor(score))
            model_dict["msid"] = log_dict["validation/msid"] * METRIC_SCALE_FROM_NAME["msid"]

        # Remove `loss` from list
        model_dict.pop("loss")
        tqdm_dict["val_loss"] = log_dict["validation/loss"]

        self.logger.experiment.add_scalars(f"models/{self.hparams.name}", model_dict, self.current_epoch)
        result = {'progress_bar': tqdm_dict, 'log': log_dict, 'val_loss': log_dict["validation/loss"]}
        return result

    def on_epoch_end(self):
        # Save images only on validation epochs
        if (self.current_epoch + 1) % self.hparams.check_val_every_n_epoch == 0:
            images, target = self.saved_batch
            output = self(images)

            N = self.hparams.num_images_to_log
            grid_input = torchvision.utils.make_grid(images[:N], nrow=int(math.sqrt(N)), normalize=True)
            grid_target = torchvision.utils.make_grid(target[:N], nrow=int(math.sqrt(N)), normalize=True)
            grid_output = torchvision.utils.make_grid(output[:N], nrow=int(math.sqrt(N)), normalize=True)

            # Concat along X axis
            final_image = torch.cat([grid_input, grid_output, grid_target], dim=2)
            self.logger.experiment.add_image(f'Validation_images', final_image, self.current_epoch)
