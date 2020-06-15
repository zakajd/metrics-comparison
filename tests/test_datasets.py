import torch
import pytest

from src.data.datasets import *


ALL_DATASETS = [
    # MNIST,
    # FashionMNIST, 
    # CIFAR10,
    # CIFAR100,
    TinyImageNet,
    DIV2K,
    Set5,
    Set14,
    Urban100,
    Manga109,
    COIL100,
    BSDS100,
    # MedicalDecathlon,
    TID2013,
]

def _test_init(dataset):
    return dataset[0]

@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_init(dataset):
    d = dataset()
    _test_init(d)
