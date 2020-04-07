# Code mostly copy-pasted from this repo
# https://github.com/seungwonpark/melgan/
from collections import OrderedDict

import numpy as np
import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pytorch_lightning as pl


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(MelGAN, self).__init__()
        self.hparams = hparams

        # Networks
        self.generator = Generator(hparams.n_mels)
        self.discriminator = NLayerDiscriminator()

        # No transformations for Mel Spectogram for now
        transforms = None 
        # Waveform -> MelSpectogram
        # self.mel = Audio2Mel(**hparams)

        self.mel = Audio2Mel(
            sampling_rate=hparams.sampling_rate,
            n_fft=hparams.n_fft,
            window_size=hparams.window_size,
            hop_size=hparams.hop_size,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            n_mels=hparams.n_mels,
            ref_level_db=hparams.ref_level_db,
            min_level_db=hparams.min_level_db,
        )

        # Cache for generated waveforms
        self.generated_waveforms = None
        self.last_waveforms = None
        self.last_melspectograms = None

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.adam_b1
        b2 = self.hparams.adam_b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def forward(self, x):
        return self.generator(x)

    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        waveforms = batch
        # self.last_waveforms = waveforms

        wf_length = waveforms.shape[-1] # Save original length of waveforms
        waveforms, melspectograms = self.mel(waveforms) # Here we get padded waveforms
        fake_waveforms = self(melspectograms)

        # Train generator
        if optimizer_idx == 0:

            with torch.no_grad():
                _, fake_melspectograms = self.mel(fake_waveforms[..., : wf_length])
                l1_loss = F.l1_loss(fake_melspectograms, melspectograms)
            
            disc_fake = self.discriminator(fake_waveforms)
            disc_real = self.discriminator(waveforms)

            g_loss = 0.0
            fake_features, score_fake = disc_fake[:-1], disc_fake[-1]
            real_features, _ = disc_real[:-1], disc_real[-1]

            # We use LSGAN loss https://arxiv.org/pdf/1611.04076v3.pdf
            g_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))

            # Additionally we match features:
            for fake_f, real_f in zip(fake_features, real_features):
                g_loss += 10. * torch.mean(torch.abs(fake_f - real_f))

            tqdm_dict = {'g_loss': g_loss, 'L1_loss' : l1_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            # Save for TensorBoard logging
            self.fake_melspectograms = fake_melspectograms

            return output

        # Train discriminator
        if optimizer_idx == 1:
            # Do 2 steps per each generator step
            d_loss = 0.0
            for _ in range(2):
                disc_fake = self.discriminator(fake_waveforms)
                disc_real = self.discriminator(waveforms)

                _, score_fake = disc_fake[:-1], disc_fake[-1]
                _, score_real = disc_real[:-1], disc_real[-1]
                
                # We use LSGAN loss https://arxiv.org/pdf/1611.04076v3.pdf
                d_loss += 0.5 * torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                d_loss += 0.5 * torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        waveforms, melspectograms = self.mel(waveforms)

        # Save for TensorBoard logging
        self.last_waveforms = waveforms

        predicted_waveforms = self(melspectograms)
        return {'val_loss': F.l1_loss(predicted_waveforms, waveforms)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def train_dataloader(self):
        # Transforms
        transforms = torchvision.transforms.Compose([
            RandomCrop1D(size=self.hparams.sample_length),
            torchaudio.transforms.Fade(
                fade_in_len=self.hparams.fade_length, 
                fade_out_len=self.hparams.fade_length)           
        ])

        return get_dataloader(
            train=True, dataset_class=AudioDataset, transforms=transforms, **vars(self.hparams)
            )

    def val_dataloader(self):
        transforms = torchvision.transforms.Compose([
            # RandomCrop1D(size=self.hparams.sample_length),
            torchaudio.transforms.Fade(
                fade_in_len=self.hparams.fade_length, 
                fade_out_len=self.hparams.fade_length)
        ])

        return get_dataloader(
            train=False, dataset_class=AudioDataset, transforms=transforms, **vars(self.hparams)
            )

    def on_epoch_end(self):
        _, melspectograms = self.mel(self.last_waveforms)
        generated_waveforms = self(melspectograms)
        # TB takes [batch, time, channel] tensor as input, so swap last two axes.
        self.logger.experiment.add_audio(
            f'generated_waveforms', generated_waveforms[0], self.current_epoch, sample_rate=self.hparams.sampling_rate)
        self.logger.experiment.add_audio(
            f'original_waveforms', self.last_waveforms[0], self.current_epoch, sample_rate=self.hparams.sampling_rate)

        # TB takes [n_channels, H, W] images, so add fake channel dimension to spectogram
        self.logger.experiment.add_image(f'original_wf_melspectogram', melspectograms[0].unsqueeze(0), self.current_epoch)
        self.logger.experiment.add_image(f'generated_wf_melspectogram', self.fake_melspectograms[0].unsqueeze(0), self.current_epoch)
