import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt


def get_aug(
    aug_type="val",
    task="denoise",
    data_mean=(0.5, 0.5, 0.5),
    data_std=(0.5, 0.5, 0.5),
    size=64):
    """
    Args:
        aug_type (str): {`val`, `test`, `light`, `medium`}
        task (str): {"denoise", "deblur"}
        data_mean (tuple): Per-channel mean for normalization
        data_std (tuple): Per-channel mean for normalization
        size (int): final size of the crop
    """

    assert aug_type in ["val", "test", "light", "medium"]

    NORM_TO_TENSOR = albu.Compose([
        albu.Normalize(mean=data_mean, std=data_std),
        albu.pytorch.ToTensorV2()
    ])

#     CROP_AUG = albu.RandomResizedCrop(size, size, scale=(0.05, 0.4))
    CROP_AUG = albu.NoOp()  # No crops for small datasets

    if task == "deblur":
        TASK_AUG = albu.OneOf([
            # albu.Blur(),
            albu.GaussianBlur(),
            # albu.MotionBlur(),
            # albu.MedianBlur(),
            # albu.GlassBlur(),
        ], p=1.0)
    elif task == "denoise":
        TASK_AUG = albu.OneOf([
            albu.GaussNoise(),
            albu.GlassBlur(),
            albu.ISONoise(),
            albu.MultiplicativeNoise()
        ], p=1.0)
    else:
        TASK_AUG = albu.NoOp()

    VAL_AUG = albu.Compose([
        # albu.CenterCrop(size, size),
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
        "val" : VAL_AUG,
        # "test" : TEST_AUG,
        "light" : LIGHT_AUG,
        "medium" : MEDIUM_AUG,
    }

    return types[aug_type]
