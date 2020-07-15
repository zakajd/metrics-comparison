import cv2
# import torch
import numpy as np
import albumentations as albu
import albumentations.pytorch as albu_pt


class GaussNoiseNoClipping(albu.GaussNoise):
    """Apply Gaussian noise without clipping to [0, 1] range.
    """
    def __init__(self, singlechannel=False, **kwargs):
        self.singlechannel = singlechannel
        super().__init__(**kwargs)

    def apply(self, img, gauss=None, **params):
        if self.singlechannel:
            return img + gauss[..., 0][..., np.newaxis]
        else:
            return img + gauss


# Normalize data to [-1, 1] range or somewhere near it
# name: (mean, std, scale)
MEAN_STD_BY_NAME = {
    "mnist": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "fashion_mnist": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "cifar10": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "cifar100": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "tinyimagenet": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    # "tinyimagenet": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # "div2k": ((0.4869, 0.4446, 0.3957), (0.0466, 0.0412, 0.0401)),
    "div2k": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "set5": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "set14": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "urban100": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "manga109": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "coil100": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "bsds100": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "medicaldecathlon": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.),
}


def get_aug(aug_type: str = "val", task: str = "denoise", dataset: str = "cifar100", size: int = 64):
    """
    Args:
        aug_type: {`val`, `test`, `light`, `medium`}
        task: {"denoise", "deblur", "sr"}
        dataset: Name of dataset to get MEAN and STD
        size: final size of the crop
    """

    assert aug_type in ["val", "test", "light", "medium"]

    # Add the same noise for all channels for single-channel images
    mean, std, max_value = MEAN_STD_BY_NAME[dataset]
    if dataset == "medicaldecathlon":
        singlechannel = True
        normalization = albu.NoOp()
        noise = GaussNoiseNoClipping(singlechannel, var_limit=0.1,)
    else:
        singlechannel = False
        normalization = albu.Normalize(mean=mean, std=std, max_pixel_value=max_value)
        noise = albu.MultiplicativeNoise(multiplier=(0.75, 1.25), per_channel=True, elementwise=True)

    NORM_TO_TENSOR = albu.Compose([
        normalization,
        albu_pt.ToTensorV2()],
        additional_targets={"mask": "image"})

    CROP_AUG = albu.Compose([
        albu.PadIfNeeded(size, size),
        albu.RandomResizedCrop(size, size, scale=(0.5, 1.)),
    ])


    if task == "deblur":
        TASK_AUG = albu.OneOf([
            albu.Blur(blur_limit=(3, 5)),
            albu.GaussianBlur(blur_limit=(3, 5)),
            # albu.MotionBlur(),
            # albu.MedianBlur(),
            # albu.GlassBlur(),
        ], p=1.0)
    elif task == "denoise":
#         TASK_AUG = noise
        TASK_AUG = albu.OneOf([
            noise,
            # albu.GaussNoise(),
            # GaussNoiseNoClipping(singlechannel, var_limit=0.1 if singlechannel else (20., 50.)),
#             albu.GlassBlur(),
#             albu.ISONoise(),
#             albu.MultiplicativeNoise()
        ], p=1.0)
    elif task == "sr":
        TASK_AUG = albu.Downscale(
            scale_min=0.5, scale_max=0.5, interpolation=cv2.INTER_CUBIC, always_apply=True)
    else:
        raise ValueError("Name of task must be in {'deblur', 'denosie', 'sr'}")

    VAL_AUG = albu.Compose([
        albu.PadIfNeeded(size, size),
        albu.CenterCrop(size, size),
        TASK_AUG,
        NORM_TO_TENSOR,
    ])

    LIGHT_AUG = albu.Compose([
        CROP_AUG,
        TASK_AUG,
        NORM_TO_TENSOR,
    ])

    MEDIUM_AUG = albu.Compose([
        albu.Flip(),
        albu.RandomRotate90(),
        CROP_AUG,
        TASK_AUG,
        NORM_TO_TENSOR
    ])

    types = {
        "val": VAL_AUG,
        "light": LIGHT_AUG,
        "medium": MEDIUM_AUG,
    }

    return types[aug_type]

