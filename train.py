import time

# import torch
import pytorch_lightning as pl
# from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.arg_parser import parse_args
# from src.datasets import get_dataloader
from src.utils import set_random_seed
from src.experiment import BaseModel


def main():
    hparams = parse_args()
    print(hparams)

    # Fix all seeds for reprodusability
    set_random_seed(hparams.seed)

    model = BaseModel(hparams)

    profiler = True  # , AdvancedProfiler()
    logger = TensorBoardLogger("logs", name=hparams.name)
    # checkpoint_callback = ModelCheckpoint(
    #     filepath='models/{epoch}-{val_loss:.2f}')

    trainer = pl.Trainer(
        logger=logger,
        gpus=hparams.device,
        benchmark=False,
        # auto_lr_find=True,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=False,  # overfit_pct=0.10,
        max_epochs=hparams.epochs,
        profiler=profiler,
        weights_summary='top',
        nb_sanity_val_steps=0,  # skip start check
        precision=32,
        amp_level='O1',
        # checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
