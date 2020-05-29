# import os
# import glob
from functools import reduce

import cv2
import h5py
# import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# from src.augmentations import get_aug
from src.utils import walk_files


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is untrasformed original image
                and input is transformed original image
        """
        img = self.data[index]
        # Add channels dimension and stack 3 identical tensors
        # Shape (H, W, channels)
        img = img.unsqueeze(2).repeat(1, 1, 3)

        # Convert to Numpy, so Albumentations can work
        img = img.numpy()

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target


class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is untrasformed original image
                and input is transformed original image
        """
        img = self.data[index]
        # Add channels dimension and stack 3 identical tensors
        # Shape (H, W, channels)
        img = img.unsqueeze(2).repeat(1, 1, 3)

        # Convert to Numpy, so Albumentations can work
        img = img.numpy()

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is untrasformed original image
                and input is transformed original image
        """
        img = self.data[index]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is untrasformed original image.
        """
        img = self.data[index]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target


class Set5(Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/Set5", train=False, transform=None):
        # assert train is False, "This dataset can be used only for validation"
        walker = walk_files(
            root, suffix=".png", prefix=True, remove_suffix=False
        )

        self.files = list(walker)
        train_size = int(len(self.files) * 0.85)
        if train:
            self.files = self.files[:train_size]
        else:
            self.files = self.files[train_size:]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target)
        """
        # Read image
        img = cv2.imread(self.files[index], cv2.IMREAD_UNCHANGED)
        # Covert to RGB and clip to [0., 1.]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target

    def __len__(self):
        return len(self.files)


class Set14(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/Set14", train=False, transform=None):
        super().__init__(root, train, transform)


class Urban100(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/Urban100", train=False, transform=None):
        super().__init__(root, train, transform)


class Manga109(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/Manga109", train=False, transform=None):
        super().__init__(root, train, transform)


class COIL100(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/coil-100", train=False, transform=None):
        super().__init__(root, train, transform)


class DIV2K(Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(self, root="datasets/", train=True, transform=None):

        root += "DIV2K_" + ('train' if train else 'valid') + "_LR_bicubic/X2"
        walker = walk_files(
            root, suffix=".png", prefix=True, remove_suffix=False
        )

        self.files = list(walker)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target)
        """
        # Read image
        img = cv2.imread(self.files[index], cv2.IMREAD_UNCHANGED)
        # Covert to RGB and clip to [0., 1.]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target

    def __len__(self):
        return len(self.files)


class BSDS100(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/BSDS100", train=False, transform=None):
        super().__init__(root, train, transform)


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(
            self, root="datasets/tiny-imagenet-200", train=True, transform=None):

        root += "/train" if train else "/val"
        walker = walk_files(
            root, suffix=".JPEG", prefix=True, remove_suffix=False
        )

        self.files = list(walker)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target)
        """
        # Read image
        img = cv2.imread(self.files[index], cv2.IMREAD_UNCHANGED)
        # Covert to RGB and clip to [0., 1.]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target

    def __len__(self):
        return len(self.files)


class MedicalDecathlon(Dataset):
    """Used to access images from MedicalDecathlon challenge
    """
    def __init__(
            self, data_path="datasets/decathlon", filename="colon.h5", train=True, transform=None):
        """
        Args:
            data_path (str): Path to folder with hp5 datasets
            filename (str): {`colon`} type of data
            train (bool): Flag to return train or val dataset
        """

        # Read all the data into memory
        with h5py.File(data_path + "/" + filename, "r") as f:
            if train:
                # Take each 10th image so that they are not highly correlated
                self.data = f['imgs_train'][::10]
            else:
                # Combine test and validation datasets
                data_testing = f['imgs_testing'][::10]
                data_validation = f['imgs_validation'][::10]
                self.data = np.concatenate((data_validation, data_testing))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # Convert (H, W) -> (H, W, 3) for transform. Cast: float64 -> float32
        img = img.repeat(3, axis=2).astype(np.float32)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            input, target = augmented["image"], augmented["mask"]
        else:
            input, target = img, img

        return input, target


def get_dataloader(datasets, transform=None, batch_size=128,
                   train=True, train_size=0.8, **kwargs):
    """Returns data from several datasets

    Args:
        datasets (list): Datset names, should be in {"mnist", "cifar10", "cifar100"}.
        batch_size (int): Size of batch generated by dataloader.
        train (bool): Return train data if `True`, validation data if `False`.
        train_size (float): Part of data used for training

    Returns:
        dataloader
    """

    # Get datasets
    all_datasets = []
    if "mnist" in datasets:
        dataset = MNIST("datasets/", train, transform)
        all_datasets.append(dataset)

    if "fashion_mnist" in datasets:
        dataset = FashionMNIST("datasets/", train, transform)
        all_datasets.append(dataset)

    if "cifar10" in datasets:
        dataset = CIFAR10("datasets/", train, transform)
        all_datasets.append(dataset)

    if "cifar100" in datasets:
        dataset = CIFAR100("datasets/", train, transform)
        all_datasets.append(dataset)

    if "tinyimagenet" in datasets:
        dataset = TinyImageNet("datasets/tiny-imagenet-200", train, transform)
        all_datasets.append(dataset)

    if "div2k" in datasets:
        dataset = DIV2K("datasets/", train, transform)
        all_datasets.append(dataset)

    if "coil100" in datasets:
        dataset = COIL100("datasets/coil-100", train, transform)
        all_datasets.append(dataset)

    if "bsds100" in datasets:
        dataset = BSDS100("datasets/BSDS100", train, transform)
        all_datasets.append(dataset)

    if "medicaldecathlon" in datasets:
        dataset = MedicalDecathlon("datasets/decathlon", "colon.h5", train, transform)
        all_datasets.append(dataset)

    #  Concat all datasets into one
    all_datasets = reduce(lambda x, y: x + y, all_datasets)

    if train:
        dataloader = DataLoader(
            all_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True)
    else:
        dataloader = DataLoader(
            all_datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    print(f"\nUsing datasets: {datasets}. {'Train' if train else 'Validation'} size: {len(all_datasets)}.")
    return dataloader
