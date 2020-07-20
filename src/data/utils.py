import os
from typing import Union, Tuple

import torch


def walk_files(root: str,
               suffix: Union[str, Tuple[str]],
               prefix: bool = True,
               remove_suffix: bool = False) -> str:
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the full path to each result, otherwise
            only returns the name of the files found (Default: ``False``)
        remove_suffix (bool, optional): If true, removes the suffix to each result defined in suffix,
            otherwise will return the result as found (Default: ``False``).
    """

    root = os.path.expanduser(root)

    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix):

                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(dirpath, f)

                yield f


def crop_patches(images: torch.Tensor, size=64, stride=32):
    """Crop input images into smaller patches
    Args:
        images: Tensor of images with shape (batch x 3 x H x W)
        size: size of a square patch
        stride: Step between patches
    """
    patches = images.data.unfold(1, 3, 3).unfold(2, size, stride).unfold(3, size, stride)
    patches = patches.reshape(-1, 3, size, size)
    return patches
