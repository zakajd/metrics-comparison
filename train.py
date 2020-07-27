import os
import sys
import yaml
import time

import torch
from loguru import logger
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb

from src.features.functional import metrics_from_list, Runner, criterion_from_list
from src.arg_parser import parse_args
from src.data import get_dataloader, get_aug
from src.features.models import MODEL_FROM_NAME
import src.features.callbacks as clb



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
    # Use print instead of logger to have alphabetic order.
    logger.info(f"Parameters used for training: {vars(hparams)}")

    # Fix all seeds for reprodusability
    pt.utils.misc.set_random_seed(hparams.seed)

    # Save config
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + "/config.yaml", "w"))

    # Get models and optimizers
    model = MODEL_FROM_NAME[hparams.model](**hparams.model_params).cuda()
    logger.info(f"Model size: {pt.utils.misc.count_parameters(model)[0] / 1e6:.02f}M")

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=hparams.weight_decay, amsgrad=True)  # Get LR from phases later

    # Get loss
    loss = criterion_from_list(hparams.criterion).cuda()

    # Init per-image metrics and add names
    metrics = metrics_from_list(hparams.metrics, reduction='mean')
    logger.info(f"Metrics: {[m.name for m in metrics]}")

    # Feature metrics are defined as a callback
    feature_clb_vgg16 = clb.FeatureMetrics(
        feature_extractor="vgg16", metric_names=hparams.feature_metrics)
    feature_clb_vgg19 = clb.FeatureMetrics(
        feature_extractor="vgg19", metric_names=hparams.feature_metrics)
    feature_clb_inception = clb.FeatureMetrics(
        feature_extractor="inception", metric_names=hparams.feature_metrics)

    # Scheduler is an advanced way of planning experiment
    sheduler = pt_clb.PhasesScheduler(hparams.phases)

    save_name = "model_{monitor}.chpn"
    # Init train loop
    runner = Runner(
        model=model,
        optimizer=optimizer,
        criterion=loss,
        callbacks=[
            pt_clb.Timer(),
            clb.ConsoleLogger(metrics=["ssim", "psnr"]),
            feature_clb_vgg16,
            feature_clb_vgg19,
            feature_clb_inception,
            clb.TensorBoard(hparams.outdir, log_every=40, num_images=2),

            # List of CheckpointSavers, one per metric
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='loss', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='psnr', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='ssim', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='ms-ssim', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='gmsd', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='ms-gmsd', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='ms-gmsdc', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='fsim', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='fsimc', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='vsi', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='mdsi', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='vifp', mode='max', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='content_vgg16', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='content_vgg19', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='content_vgg16_ap', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='content_vgg19_ap', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='style_vgg16', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='style_vgg19', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='lpips', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='dists', mode='min', verbose=False),
            clb.CheckpointSaver(
                hparams.outdir, save_name=save_name, monitor='brisque', mode='min', verbose=False),
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
    runner.fit(
        train_loader,
        epochs=sheduler.tot_epochs,
        val_loader=val_loader,
        steps_per_epoch=2 if hparams.debug else None,
        val_steps=2 if hparams.debug else None,
    )

    logger.info("Finished training!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
