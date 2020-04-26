# import os
import math
from collections import OrderedDict


import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
# import photosynthesis_metrics as pm


from src.augmentations import get_aug
from src.datasets import get_dataloader
from src.modules import Identity, MODEL_FROM_NAME
from src.utils import METRIC_FROM_NAME, METRIC_SCALE_FROM_NAME


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        hparams.model_params["upsample"] = True if hparams.task == "sr" else False
        self.model = MODEL_FROM_NAME[hparams.model](**hparams.model_params)

        self.feature_extractor = torchvision.models.__dict__[self.hparams.feature_extractor](
            pretrained=True
        )

        # Hack to get intermidiate features for ResNet model
        self.feature_extractor.fc = Identity()

        self.validation_features = None
        self.criterion = torch.nn.MSELoss() 
        #  F.mse_loss(prediction, target)

        # Init per-image metrics
        self.metric_names = hparams.metrics[::2]
        self.metrics = [
            METRIC_FROM_NAME[name](**kwargs) for name, kwargs in zip(hparams.metrics[::2], hparams.metrics[1::2])
        ]

        print("Metrics", self.metrics)
        # Init feature metrics
        self.feat_metric_names = hparams.feature_metrics[::2]

        print(self.feat_metric_names)
        self.feat_metrics = [
            METRIC_FROM_NAME[name](**kwargs) for name, kwargs in zip(hparams.feature_metrics[::2], hparams.feature_metrics[1::2])
        ]
        print("Feature metrics", self.feat_metrics)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch

        # Save for logging
        self.last_batch = batch
        prediction = self(input)

        loss = self.criterion(prediction, target)

        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'loss': loss
        })

        return output

    def validation_step(self, batch, batch_idx):
        input, target = batch
        prediction = self(input)

        with torch.no_grad():
            # loss_val = F.mse_loss(prediction, target)
            loss_val = self.criterion(prediction, target)

            output = OrderedDict({'loss': loss_val})

            # Compute metrics
            for i, name in enumerate(self.metric_names):
                output[name] = self.metrics[i](prediction, target)

            input_features = self.feature_extractor(input)

            if self.validation_features is None:
                target_features = self.feature_extractor(target)
            else:
                target_features = None

        output['input_features'] = input_features
        output['target_features'] = target_features
        return output

    def validation_epoch_end(self, outputs):

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
            if len(outputs) == 0:
                print(outputs)
            log_dict["validation/" + metric_name] = metric_total / len(outputs)
            model_dict[metric_name] = log_dict["validation/" + metric_name] * METRIC_SCALE_FROM_NAME[metric_name]

        self.metric_names.pop()  # Remove `loss` from this list
        model_dict.pop("loss")

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

        tqdm_dict["val_loss"] = log_dict["validation/loss"]

        self.logger.experiment.add_scalars(f"models/{self.hparams.name}", model_dict, self.current_epoch)
        result = {'progress_bar': tqdm_dict, 'log': log_dict, 'val_loss': log_dict["validation/loss"]}
        return result

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Transforms
        hp = self.hparams
        transform = get_aug(
            aug_type=hp.aug_type,
            task=hp.task,
            dataset=hp.datasets[0],
            size=hp.size
        )

        # Normalize and convert to tensor

        train_loader = get_dataloader(
            datasets=hp.datasets,
            train=True,
            transform=transform,
            batch_size=hp.batch_size
        )

        return train_loader

    def val_dataloader(self):
        hp = self.hparams

        # Transforms
        transform = get_aug(
            aug_type=hp.aug_type,
            task=hp.task,
            dataset=hp.datasets[0],
            size=hp.size
        )

        # Normalize and convert to tensor

        val_loader = get_dataloader(
            datasets=hp.datasets,
            train=False,
            transform=transform,
            batch_size=hp.batch_size

        )
        return val_loader

    def on_epoch_end(self):
        images, target = self.last_batch
        output = self(images)

        # Upscale images by bilinear interpolation to get the same image size.
        if self.hparams.task == "sr":
            print(f"Before interpolation: images {images.shape}, targets {target.shape}, output {output.shape} ")
            images = F.interpolate(images, size=target.shape[-2:], mode="bilinear")
            print(f"After interpolation: images {images.shape}, targets {target.shape}, output {output.shape} ")

        N = self.hparams.num_images_to_log
        grid_input = torchvision.utils.make_grid(images[:N], nrow=int(math.sqrt(N)), normalize=True)
        grid_target = torchvision.utils.make_grid(target[:N], nrow=int(math.sqrt(N)), normalize=True)
        grid_output = torchvision.utils.make_grid(output[:N], nrow=int(math.sqrt(N)), normalize=True)

        # Concat along X axis
        final_image = torch.cat([grid_input, grid_output, grid_target], dim=2)
        self.logger.experiment.add_image(f'Validation_images', final_image, self.current_epoch)

