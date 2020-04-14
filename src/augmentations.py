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


def get_aug(
    aug_type="val",
    task="denoise",
    data_mean=(0.0 , 0.0, 0.0),
    data_std=(1.0, 1.0, 1.0),
    size=64):
    """
    Args:
        aug_type (str): {`val`, `test`, `light`, `medium`}
        task (str): {"denoise", "deblur", "sr"}
        data_mean (tuple): Per-channel mean for normalization
        data_std (tuple): Per-channel mean for normalization
        size (int): final size of the crop
    """

    assert aug_type in ["val", "test", "light", "medium"]

    NORM_TO_TENSOR = albu.Compose([
        albu.Normalize(mean=data_mean, std=data_std, max_pixel_value=255.0),
        albu.pytorch.ToTensorV2()],
        additional_targets={"mask" : "image"})

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
            albu.ISONoise(),
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
        "val" : VAL_AUG,
        # "test" : TEST_AUG,
        "light" : LIGHT_AUG,
        "medium" : MEDIUM_AUG,
    }

    return types[aug_type]
