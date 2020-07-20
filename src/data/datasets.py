# import os
# import glob
from functools import reduce

import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd
import torchvision
from loguru import logger
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
import albumentations.pytorch as albu_pt


from src.data.utils import walk_files

class DistortionSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements with same distortion type

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, dist_type=None):
        self.data_source = data_source
        if dist_type is not None:
            self.data_source.df = data_source.df[data_source.df['dist_type'] == dist_type]
            self.data_source.scores = self.data_source.df['score'].to_numpy()
        else:
            self.data_source = data_source
            
    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source.df)

class TID2013(torch.utils.data.Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
        
    Returns:
        distorted: image with some kind of distortion
        reference: image without distortions
        score: MOS score for this pair of images
    """
    _filename = "mos_with_names.txt"
    
    def __init__(self, root="data/raw/tid2013", transform=None):
        self.df = pd.read_csv(
            os.path.join(root, self._filename),
            sep=' ',
            names=['score', 'dist_img'],
            header=None
        )
        

        self.df["ref_img"] = self.df["dist_img"].apply(lambda x: (x[:3] + x[-4:]).upper())
        self.df['dist_type'] = self.df['dist_img'].apply(lambda x: x[4:-4])
        self.scores = self.df['score'].to_numpy()
        self.root = root
        
        if transform is None:
            self.transform = albu_pt.ToTensorV2()
        else:
            self.transform = transform

    def __getitem__(self, index):
        distorted_path = os.path.join(self.root, "distorted_images", self.df.iloc[index][1])
        reference_path = os.path.join(self.root, "reference_images", self.df.iloc[index][2])
        score = self.scores[index]
        
        # Load image and ref
        distorted = cv2.imread(distorted_path, cv2.IMREAD_UNCHANGED)
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        distorted = self.transform(image=distorted)["image"]

        reference = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)   
        reference = self.transform(image=reference)["image"]
        
        return distorted, reference, score

    def __len__(self):
        return len(self.df)


class KADID10k(torch.utils.data.Dataset):
    """
    Total length = 
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
        
    Returns:
        distorted: 25 images with fixed distortion type and level
        reference: 25 original images
    """
    _filename = "dmos.csv"
    
    def __init__(
        self, root="data/raw/kadid10k", transform=None):
        
        self.images_root = os.path.join(root, "images")
    
        # Read file mith DMOS and names
        self.df = pd.read_csv(os.path.join(root, self._filename))
        self.df.rename(columns={"dmos": "score"}, inplace=True)
        self.scores = self.df["score"].to_numpy()
        self.df['dist_type'] = self.df['dist_img'].apply(lambda x: x[4:-4])
        
        if transform is None:
            self.transform = albu_pt.ToTensorV2()
        else:
            self.transform = transform

    def __getitem__(self, index):
        distorted_path = os.path.join(self.images_root, self.df.iloc[index][0])
        reference_path = os.path.join(self.images_root, self.df.iloc[index][1])
        score = self.scores[index]
        
        # Load image and ref
        distorted = cv2.imread(distorted_path, cv2.IMREAD_UNCHANGED)
        distorted = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
        distorted = self.transform(image=distorted)["image"]

        reference = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED)
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)   
        reference = self.transform(image=reference)["image"]
        
        return distorted, reference, score
    
    def __len__(self):
        return len(self.df)

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


class Set5(torch.utils.data.Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "Set5"
    def __init__(
            self, root="data/raw", train=False, transform=None):
        assert train is False, "This dataset can be used only for validation"
        root = os.path.join(root, self._folder)
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


class Set14(Set5):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "Set14"
    def __init__(
            self, root="data/raw", train=False, transform=None):
        super().__init__(root, train, transform)


class Urban100(torch.utils.data.Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "Urban100"
    def __init__(self, root="data/raw", train=False, transform=None):
        root = os.path.join(root, _folder)

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


class Manga109(Urban100):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "Manga109"
    def __init__(
            self, root="data/raw", train=False, transform=None):
        super().__init__(root, train, transform)


class COIL100(Urban100):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "coil-100"
    def __init__(
            self, root="data/raw/coil-100", train=False, transform=None):
        super().__init__(root, train, transform)


class DIV2K(torch.utils.data.Dataset):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    def __init__(self, root="data/raw", train=True, transform=None):

        root += "/DIV2K_" + ('train' if train else 'valid') + "_LR_bicubic/X2"
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


class BSDS100(Urban100):
    """
    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "BSDS100"
    def __init__(
            self, root="data/raw", train=False, transform=None):
        super().__init__(root, train, transform)


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Args:
        root (str) – Root directory path.
        train (bool): Flag to return train if True and validation if False
        transform (callable) – A function/transform that takes in the input and transforms it.
    """
    _folder = "tiny-imagenet-200"
    def __init__(
            self, root="data/raw", train=True, transform=None):
        root = os.path.join(root, self._folder, "train" if train else "val")
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


class MedicalDecathlon(torch.utils.data.Dataset):
    """Used to access images from MedicalDecathlon challenge
    """
    def __init__(
            self, data_path="data/raw/decathlon", filename="colon.h5", train=True, transform=None):
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



def get_dataloader(
    dataset: str, transform=None, batch_size: int = 128, train: bool = True, train_size: float = 0.8, **kwargs):
    """Returns data from several datasets

    Args:
        dataset: Datset name
        transform: Albumentations transform
        batch_size: Size of batch generated by dataloader.
        train: Return train data if `True`, validation data if `False`.
        train_size: Part of data used for training

    Returns:
        dataloader
    """

    # Get dataset
    dataset = DATASET_FROM_NAME[dataset](train=train, transform=transform)

    if "mnist" in dataset:
        dataset = MNIST("data/raw/", train, transform)
        all_datasets.append(dataset)



    if "medicaldecathlon" in datasets:
        dataset = MedicalDecathlon("data/raw/decathlon", "colon.h5", train, transform)
        all_datasets.append(dataset)

    #  Concat all datasets into one
    all_datasets = reduce(lambda x, y: x + y, all_datasets)

    if train:
        dataloader = torch.utils.data.DataLoader(
            all_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            all_datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    logger.info(f"\nUsing datasets: {datasets}. {'Train' if train else 'Validation'} size: {len(all_datasets)}.")
    return dataloader

DATASET_FROM_NAME = {
    "mnist": MNIST,
    "fashion_mnist": FashionMNIST,
    "cifa10": CIFAR10,
    "cifar100": CIFAR100,
    "tinyimagenet": TinyImageNet,
    "div2k": DIV2K,
    "coil100": COIL100,
    "bsds100": BSDS100,
    "set5": Set5,
    "set14": Set14,
    "medicaldecathlon": MedicalDecathlon,
}