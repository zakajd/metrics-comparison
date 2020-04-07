import time


import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.arg_parser import parse_args
from src.melgan import MelGAN
from src.datasets import get_dataloader
from src.utils import set_random_seed



def main():
    hparams = parse_args()
    print(hparams)

    # Fix all seeds for reprodusability
    set_random_seed(hparams.seed)  

    vocoder = MelGAN(hparams)

    profiler = True # , AdvancedProfiler()
    logger = TensorBoardLogger("logs", name="MelGAN")
    checkpoint_callback = ModelCheckpoint(
        filepath='models/{epoch}-{val_loss:.2f}')
    # # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(
        logger=logger, gpus=hparams.device, benchmark=True, 
        check_val_every_n_epoch=2, fast_dev_run=False, #overfit_pct=0.01,
        log_gpu_memory='min_max',max_epochs=40, profiler=profiler, 
        precision=32, amp_level='O1', checkpoint_callback=checkpoint_callback)

    # overfit_pct=0.05    
    trainer.fit(vocoder)   


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")