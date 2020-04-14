import os
import math
from collections import OrderedDict


import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import photosynthesis_metrics as pm


from src.augmentations import get_aug
from src.datasets import get_dataloader
from src.modules import Identity, MODEL_FROM_NAME



class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = MODEL_FROM_NAME[hparams.model](**hparams.model_params)

        self.feature_extractor = torchvision.models.__dict__[self.hparams.feature_extractor](
            pretrained=True
        )

        # Hack to get intermidiate features for ResNet model
        self.feature_extractor.fc = Identity()

        self.validation_features = None
        # self.criterion = torch.nn.MSELoss()

        # Per-image metrics
        # self.ms_ssim = pm.MultiScaleSSIMLoss(kernel_size=3)
        self.ssim = pm.SSIMLoss(kernel_size=3)

        # Distribution metrics
        self.msid = pm.MSID()
        self.fid = pm.FID()
        self.kid = pm.KID()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch

        # Save for logging
        self.last_batch = batch

        prediction = self(input)
        # loss = F.mse_loss(prediction, target)
        loss = F.l1_loss(prediction, target)

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
            loss_val = F.l1_loss(prediction, target)

            # Compute metrics
            mse = torch.mean((prediction - target) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ssim_score = self.ssim(prediction, target, data_range=2.)
            # ms_ssim_score = self.ms_ssim(prediction, target, data_range=2.)

            input_features = self.feature_extractor(input)

            if self.validation_features is None:
                target_features = self.feature_extractor(target)
            else:
                target_features = None

        # output = OrderedDict({
        #     'val_loss': loss_val,
        #     'val_mse': mse,
        #     'val_psnr': psnr,
        #     'val_ssim': ssim_score,
        #     # 'val_ms_ssim': ms_ssim_score,
        #     'input_features': input_features,
        #     'target_features': target_features
        # })

        output = OrderedDict({
            'loss': loss_val,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_score,
            # 'val_ms_ssim': ms_ssim_score,
            'input_features': input_features,
            'target_features': target_features
        })

        return output

    def validation_epoch_end(self, outputs):

        tqdm_dict = {}
        log_dict = {}

        # Reduce per-image metrics
        for metric_name in ["loss", "mse", "psnr", "ssim"]: # val_ms_ssim
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                metric_total += metric_value

            log_dict["validation/" + metric_name] = metric_total / len(outputs)

        # Collect computed image features into a vector of size (val_size, feat_size)
        all_input_features = [out["input_features"] for out in outputs]
        input_features = torch.cat(all_input_features, dim=0)

        if self.validation_features is None:
            all_target_features = [out["target_features"] for out in outputs]
            self.validation_features = torch.cat(all_target_features, dim=0)
        
        msid_score = []
        for _ in range(self.hparams.compute_metrics_repeat):
            msid_score.append(self.msid(input_features.cpu(), self.validation_features.cpu()))

        msid_score = torch.mean(torch.tensor(msid_score))
        kid_score = self.kid(input_features.cpu(), self.validation_features.cpu())
        fid_score = self.fid(input_features.cpu(), self.validation_features.cpu())
    
        
        log_dict["validation/msid"] = torch.tensor(msid_score)
        log_dict["validation/kid"] = torch.tensor(kid_score)
        log_dict["validation/fid"] = torch.tensor(fid_score)

        tqdm_dict["val_loss"] = log_dict["validation/loss"]

        result = {'progress_bar': tqdm_dict, 'log': log_dict, 'val_loss': log_dict["validation/loss"]}
        return result

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=self.hparams.momentum,
        #     weight_decay=self.hparams.weight_decay
        # )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer] #, [scheduler]

    def train_dataloader(self):
        # Transforms
        hp = self.hparams
        transform = get_aug(
            hp.aug_type, hp.task, hp.data_mean, hp.data_std, hp.size
        )

        # Normalize and convert to tensor
        target_transform = get_aug(
            "val", None, hp.data_mean, hp.data_std, hp.size
        )

        train_loader = get_dataloader(
            train=True,
            transform=transform,
            target_transform=target_transform,
            **vars(self.hparams)
        )

        return train_loader

    def val_dataloader(self):
        hp = self.hparams
        # Transforms
        transform = get_aug(
            hp.aug_type, hp.task, hp.data_mean, hp.data_std, hp.size
        )

        # Normalize and convert to tensor
        target_transform = get_aug(
            "val", None, hp.data_mean, hp.data_std, hp.size
        )

        val_loader = get_dataloader(
            train=False,
            transform=transform,
            target_transform=target_transform,
            **vars(self.hparams)
        )
        return val_loader

    def on_epoch_end(self):
        images, target = self.last_batch
        output = self(images)

        # Upscale images by bilinear interpolation to get the same image size. 
        if self.hparams.task == "sr":
            images = F.interpolate(images, target.shape[:-2]), mode="bilinear")
        N = self.hparams.num_images_to_log
        grid_input = torchvision.utils.make_grid(images[:N], nrow=int(math.sqrt(N)))
        grid_target = torchvision.utils.make_grid(target[:N], nrow=int(math.sqrt(N)))
        grid_output = torchvision.utils.make_grid(output[:N], nrow=int(math.sqrt(N)))

        final_image = torch.cat([grid_input, grid_output, grid_target], dim=2)
        self.logger.experiment.add_image(f'Validation mages', final_image, self.current_epoch)

