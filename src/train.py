import time

import torch
from loguru import logger
import pytorch_tools as pt

from src.arg_parser import parse_args
# from src.datasets import get_dataloader
# from src.augmentations import get_aug
from src.losses import GeneratorWGAN, DiscriminatorWGAN
from src.models import MODEL_FROM_NAME, Discriminator
from src.utils import METRIC_FROM_NAME, EXTRACTOR_FROM_NAME  # SumOfLosses, WeightedLoss
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

    ## Save config
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + "/config.yaml", "w"))

    # Get models and optimizers
    generator = MODEL_FROM_NAME[hparams.model](**hparams.model_params).cuda()
    discriminator = Discriminator().cuda()
    logger.info(f"Generator size: {pt.utils.misc.count_parameters(generator)[0] / 1e6:.02f}M")
    logger.info(f"Discriminator size: {pt.utils.misc.count_parameters(generator)[0] / 1e6:.02f}M")

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), weight_decay=self.hparams.weight_decay, amsgrad=True)  # Get LR from phases later
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), weight_decay=self.hparams.weight_decay, amsgrad=True)  # Get LR from phases later

    # Get loss
    generator_loss = GeneratorWGAN() # !!! Init params
    discriminator_loss = DiscriminatorWGAN() # !!! Init params

    # Define feature extractor
    feature_extractor = EXTRACTOR_FROM_NAME[hparams.feature_extractor]

    # Get metrics


    # Scheduler is an advanced way of planning experiment
    sheduler = PhasesScheduler(hparams.phases)

    # Init train loop
    trainer = Trainer(
        models=[generator, discriminator],
        optimizers=[generator_optimizer, discriminator_optimizer],
        criterions=[generator_loss, discriminator_loss],
        callbacks=[
            clbs.Timer(),
            clbs.ConsoleLogger(),
            # clbs.FileLogger(hparams.outdir, logger=logger),
            clbs.TensorBoard(hparams.outdir, log_every=40, num_images=2),

            # List of CheckpointSaver, one per metric
            clbs.CheckpointSaver(hparams.outdir, save_name=f"model_{fold}.chpn"),
            sheduler,
        ],
        metrics=[
            # pt.metrics.Accuracy(),
            bce_loss,
        ],
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
        val_loader=val_loader,
        steps_per_epoch=2 if hparams.debug else None,
        val_steps=2 if hparams.debug else None,
    )

    logger.info("Finished training!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
