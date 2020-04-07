import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt


def get_aug(aug_type="val", task="denoise", size=256):
    """
    Args:
        aug_type (str): {`val`, `test`, `light`, `medium`, `hard`}
        task (str): {"denoise", "deblur"}
        size (int): final size of the crop
    """

    assert aug_type in  ["val", "test", "light", "medium", "hard"]

    NORM_TO_TENSOR = albu.Compose([albu.Normalize(), ToTensor()])
    if task == "deblur":
        TASK = albu.OneOf([
            albu.Blur(),
            albu.GaussianBlur(),
            albu.MotionBlur(),
            albu.MedianBlur(),
            albu.GlassBlur(),
            ])

    TASK = 
    VAL_AUG = albu.Compose([albu.CenterCrop(size, size), NORM_TO_TENSOR,])

    TEST_AUG = albu.Compose(
        [
            albu.Resize(size, size),
            NORM_TO_TENSOR,
        ]
    )
    LIGHT_AUG = albu.Compose([CROP_AUG, albu.Flip(), albu.RandomRotate90(), NORM_TO_TENSOR,])

    MEDIUM_AUG = albu.Compose(
        [
            albu.RandomResizedCrop(size, size, scale=(0.05, 0.4)),
            albu.Flip(),
            albu.ShiftScaleRotate(),  # border_mode=cv2.BORDER_CONSTANT
            # Add occasion blur/sharpening
            albu.GaussianBlur(),
            albu.GaussNoise(),
            albu.Normalize(), 
            albu.pytorch.ToTensorV2()
        ]
    )
    
    HARD_AUG = albu.Compose(
        [   
            CROP_AUG,
            albu.RandomRotate90(),
            albu.Transpose(),
            albu.RandomGridShuffle(p=0.2),
            albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.2),
            albu.ElasticTransform(alpha_affine=5, p=0.2),
            # Add occasion blur
            albu.OneOf([albu.GaussianBlur(), albu.GaussNoise(), albu.IAAAdditiveGaussianNoise(), albu.NoOp()]),
            # D4 Augmentations
            albu.OneOf([albu.CoarseDropout(), albu.NoOp()]),
            # Spatial-preserving augmentations:
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(brightness_by_max=True),
                    albu.CLAHE(),
                    albu.HueSaturationValue(),
                    albu.RGBShift(),
                    albu.RandomGamma(),
                    albu.NoOp(),
                ]
            ),
            # Weather effects
            albu.OneOf([albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), albu.NoOp()]),
            NORM_TO_TENSOR,
        ]
    )
    

    types = {
        "val" : VAL_AUG,
        "test" : TEST_AUG,
        "light" : LIGHT_AUG,
        "medium" : MEDIUM_AUG,
        "hard": HARD_AUG,
    }

    return types[aug_type]