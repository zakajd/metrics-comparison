import os
import sys
import yaml
import time

import torch
from loguru import logger
import pytorch_tools as pt

from src.arg_parser import parse_args
from src.data import get_dataloader, get_aug
from src.modules.losses import GeneratorWGAN, DiscriminatorWGAN
from src.modules.models import MODEL_FROM_NAME, Discriminator
from src.utils import METRIC_FROM_NAME
from src.trainer import GANTrainer
import src.trainer.callbacks as clbs


def main():

    # Get config for this run
    hparams = parse_args()

    # Setup logger
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm]} - {message}"},
            {"sink": f"{hparams.outdir}/logs.txt", "format": "{time:[MM-DD HH:mm]} - {message}"},
        ],
    }
    logger.configure(**config)
    logger.info(f"Parameters used for training: {vars(hparams)}")

    # Fix all seeds for reprodusability
    pt.utils.misc.set_random_seed(hparams.seed)

    # Save config
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + "/config.yaml", "w"))

    # Get models and optimizers
    generator = MODEL_FROM_NAME[hparams.model](**hparams.model_params).cuda()
    discriminator = Discriminator().cuda()
    logger.info(f"Generator size: {pt.utils.misc.count_parameters(generator)[0] / 1e6:.02f}M")
    logger.info(f"Discriminator size: {pt.utils.misc.count_parameters(generator)[0] / 1e6:.02f}M")

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), weight_decay=hparams.weight_decay, amsgrad=True)  # Get LR from phases later
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), weight_decay=hparams.weight_decay, amsgrad=True)  # Get LR from phases later

    # Get loss
    generator_loss = GeneratorWGAN()  # !!! Init params
    discriminator_loss = DiscriminatorWGAN()  # !!! Init params

    # Init per-image metrics and add names
    metric_names = hparams.metrics[::2]
    metrics = [
        METRIC_FROM_NAME[name](**kwargs) for name, kwargs in zip(hparams.metrics[::2], hparams.metrics[1::2])
    ]
    for metric, name in zip(metrics, metric_names):
        metric.name = name
    logger.info(f"Metrics {metrics}")

    # Feature metrics are defined as a callback
    feature_clb_vgg16 = clbs.FeatureMetrics(feature_extractor="vgg16", metrics=hparams.feature_metrics)
    feature_clb_vgg19 = clbs.FeatureMetrics(feature_extractor="vgg19", metrics=hparams.feature_metrics),
    feature_clb_inception = clbs.FeatureMetrics(feature_extractor="inception", metrics=hparams.feature_metrics),

    # Scheduler is an advanced way of planning experiment
    sheduler = clbs.PhasesScheduler(hparams.phases)

    # Init train loop
    trainer = GANTrainer(
        models=[generator, discriminator],
        optimizers=[generator_optimizer, discriminator_optimizer],
        criterions=[generator_loss, discriminator_loss],
        callbacks=[
            clbs.Timer(),
            clbs.ConsoleLogger(),
            feature_clb_vgg16,
            feature_clb_vgg19,
            feature_clb_inception,
            clbs.TensorBoard(hparams.outdir, log_every=40, num_images=2),

            # List of CheckpointSaver, one per metric
            clbs.CheckpointSaver(hparams.outdir, save_name=f"model_{monitor}_{ep}.chpn", monitor='loss', minimize=True),
            sheduler,
        ],
        metrics=metrics,
    )

    # Get dataloaders
    transform = get_aug(
        aug_type=hparams.aug_type, task=hparams.task, dataset=hparams.train_dataset, size=hparams.size)
    train_loader = get_dataloader(
        dataset=hparams.train_dataset, train=True, transform=transform, batch_size=hparams.batch_size)

    transform = get_aug(
        aug_type="val", task=hparams.task, dataset=hparams.val_dataset, size=hparams.size)
    val_loader = get_dataloader(
        dataset=hparams.val_dataset, train=False, transform=transform, batch_size=hparams.batch_size)

    # Train
    trainer.fit(
        train_loader,
        val_loader=val_loader,
        steps_per_epoch=2 if hparams.debug else None,
        val_steps=2 if hparams.debug else None,
    )

    logger.info("Finished training!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
