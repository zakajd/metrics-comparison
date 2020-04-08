import os
from functools import partial, reduce


import torch
from torchvision.transforms import Compose
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

from src.augmentations import get_aug

class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is untrasformed original image
        """
        img = self.data[index]
        
        img = cv2.imread(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(img)
        else:
            target = img
            
        return img, target


class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is untrasformed original image.
        """
        img = self.data[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(img)
        else:
            target = img
            
        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is untrasformed original image.
        """
        img = self.data[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(img)
        else:
            target = img

        return img, target


# class TinyImageNet(Dataset):
#     r"""Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

#     Args:
#         root (str): Root directory including `train`, `test` and `val` subdirectories.
#         split (str): Indicating which split to return as a data set. In {`train`, `test`, `val`}
#     transform (torchvision.transforms): A (series) of valid transformation(s).
#     in_memory (bool): Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
#     """

#     _ext = 'JPEG'

#     _class_info = 'wnids.txt'
#     _class_annotation = 'words.txt'
#     VAL_ANNOTATION_FILE = 'val_annotations.txt'


#     def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=True):
#         self.root = os.path.expanduser(root)
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.in_memory = in_memory
#         self.split_dir = os.path.join(root, self.split)
#         self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
#         self.labels = {}  # fname - label number mapping
#         self.images = []  # used for in-memory processing

#         # build class label - number mapping
#         with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
#             self.label_texts = sorted([text.strip() for text in fp.readlines()])
#         self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

#         if self.split == 'train':
#             for label_text, i in self.label_text_to_number.items():
#                 for cnt in range(NUM_IMAGES_PER_CLASS):
#                     self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
#         elif self.split == 'val':
#             with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
#                 for line in fp.readlines():
#                     terms = line.split('\t')
#                     file_name, label_text = terms[0], terms[1]
#                     self.labels[file_name] = self.label_text_to_number[label_text]

#         # read all images into torch tensor in memory to minimize disk IO overhead
#         if self.in_memory:
#             self.images = [self.read_image(path) for path in self.image_paths]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         file_path = self.image_paths[index]

#         if self.in_memory:
#             img = self.images[index]
#         else:
#             img = self.read_image(file_path)

#         if self.split == 'test':
#             return img
#         else:
#             # file_name = file_path.split('/')[-1]
#             return img, self.labels[os.path.basename(file_path)]

#     def read_image(self, path):
#         img = Image.open(path)
#         return self.transform(img) if self.transform else img


def get_dataloader(
    datasets,
    transforms=None,
    batch_size=16, 
    train=True,
    train_size=0.8,
    **kwargs
    ):
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
        # dataset = dataset_class("datasets/cmu_arctic", transforms, resample, sampling_rate, **kwargs)

        # train_length=int(len(dataset) * train_size)
        # val_length=len(dataset) - train_length
        # train_d, val_d = random_split(dataset, (train_length, val_length))

        # all_datasets.append(train_d if train else val_d)
 
    # Concat all datasets into one
    all_datasets = reduce(lambda x, y: x + y, all_datasets)

    # without `drop_last` last batch consists of 1 element and BN fails TODO(zakajd): Remove this ???
    if train:
        dataloader = DataLoader(
            all_datasets, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8, 
            drop_last=True, 
            pin_memory=True)
    else:
        dataloader = DataLoader(
            all_datasets, 
            batch_size=1,
            shuffle=False, 
            num_workers=8,
            pin_memory=True)


    print(f"\nUsing datasets: {datasets}. {'Train' if train else 'Validation'} size: {len(all_datasets)}.")
    return dataloader