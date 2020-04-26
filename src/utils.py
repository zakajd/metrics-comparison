import os
import random
from typing import Tuple, Union

import torch
import numpy as np
import photosynthesis_metrics as pm


from src.losses import StyleLoss, ContentLoss, PSNR


def set_random_seed(seed):
    """Fixes all possible seeds
    Args:
        seed (int): Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


METRIC_FROM_NAME = {
    "ssim": pm.SSIMLoss,
    "ms-ssim": pm.MultiScaleSSIMLoss,
    "msid": pm.MSID,
    "fid": pm.FID,
    "kid": pm.KID,
    "content": ContentLoss,
    "style": StyleLoss,
    "tv": pm.TVLoss,
    "psnr": PSNR
}

#  Try to make all metrics have scale ~10
METRIC_SCALE_FROM_NAME = {
    "ssim": 10.,
    "ms-ssim": 10.,
    "msid": 2.,
    "fid": 0.1,
    "kid": 2.,
    "content": 15.,
    "style": 1e-5,
    "tv": 2.,
    "psnr": 0.3,
    "loss": 1.  # not used
}