import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt


class Downscale(albu.Downscale):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, image, scale, interpolation, **params):
        downscaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        return downscaled


# Normalize data to [-1, 1] range or somewhere near it
# name: (mean, std, scale)
MEAN_STD_BY_NAME = {
    "mnist": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
    "fashionmnist": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 255.),
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


def get_aug(aug_type="val", task="denoise", dataset="cifar100", size=64):
    """
    Args:
        aug_type (str): {`val`, `test`, `light`, `medium`}
        task (str): {"denoise", "deblur", "sr"}
        dataset (str): Name of dataset to get MEAN and STD
        size (int): final size of the crop
    """

    assert aug_type in ["val", "test", "light", "medium"]

    mean, std, max_value = MEAN_STD_BY_NAME[dataset]
    NORM_TO_TENSOR = albu.Compose([
        albu.NoOp() if dataset == "medicaldecathlon" else albu.Normalize(mean=mean, std=std, max_pixel_value=max_value),
        #  albu.Normalize(mean=mean, std=std, max_pixel_value=max_value),
        albu_pt.ToTensorV2()],
        additional_targets={"mask": "image"})

    CROP_AUG = albu.Compose([
        albu.PadIfNeeded(size, size),
        albu.RandomResizedCrop(size, size, scale=(0.5, 1.)),
    ])
    # CROP_AUG = albu.NoOp()  # No crops for small datasets

    if task == "deblur":
        TASK_AUG = albu.OneOf([
            albu.Blur(blur_limit=(3, 5)),
            albu.GaussianBlur(blur_limit=(3, 5)),
            # albu.MotionBlur(),
            # albu.MedianBlur(),
            # albu.GlassBlur(),
        ], p=1.0)
    elif task == "denoise":
        TASK_AUG = albu.OneOf([
            albu.GaussNoise(),
            # albu.GlassBlur(),
            albu.NoOp() if dataset == "medicaldecathlon" else albu.ISONoise(),
            # albu.MultiplicativeNoise()
        ], p=1.0)
    elif task == "sr":
        TASK_AUG = albu.OneOf([
            Downscale(scale_min=0.5, scale_max=0.5)
        ], p=1.0)
    else:
        TASK_AUG = albu.NoOp()

    VAL_AUG = albu.Compose([
        albu.PadIfNeeded(size, size),
        albu.CenterCrop(size, size),
        NORM_TO_TENSOR,
    ])

    # TEST_AUG = albu.Compose([
    #     albu.Resize(size, size),
    #     NORM_TO_TENSOR
    # ])

    LIGHT_AUG = albu.Compose([
        CROP_AUG,
        TASK_AUG,
        NORM_TO_TENSOR,
    ])

    MEDIUM_AUG = albu.Compose([
        CROP_AUG,
        TASK_AUG,
        NORM_TO_TENSOR
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
        # "test": TEST_AUG,
        "light": LIGHT_AUG,
        "medium": MEDIUM_AUG,
    }

    return types[aug_type]
