import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.arg_parser import parse_args
from src.datasets import get_dataloader
from src.utils import set_random_seed
from src.experiment import BaseModel


def main():
    hparams = parse_args()
    print(hparams)

    # Fix all seeds for reprodusability
    set_random_seed(hparams.seed)

    model = BaseModel(hparams)

    profiler = True  # , AdvancedProfiler()
    logger = TensorBoardLogger("logs", name="MNIST")
    # checkpoint_callback = ModelCheckpoint(
    #     filepath='models/{epoch}-{val_loss:.2f}')

    trainer = pl.Trainer(
        logger=logger, gpus=hparams.device, benchmark=True,
        check_val_every_n_epoch=5, fast_dev_run=True,  # overfit_pct=0.01,
        max_epochs=hparams.epochs, profiler=profiler,
        precision=32, amp_level='O1',)  # checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
