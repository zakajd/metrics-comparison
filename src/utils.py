import torchvision

from src.features.losses import MSELoss, L1Loss, MultiScaleSSIMLoss
from src.features.wrappers import InceptionV3Wrapper

EXTRACTOR_FROM_NAME = {
    "vgg16": torchvision.models.vgg16(pretrained=True, progress=False).features,
    "vgg19": torchvision.models.vgg19(pretrained=True, progress=False).features,
    "inception": InceptionV3Wrapper(resize_input=False, use_fid_inception=True, normalize_input=True),
}


# All this losses expect raw logits as inputs
LOSS_FROM_NAME = {
    "l1": L1Loss(),
    "l2": MSELoss(),
    "ms-ssim": MultiScaleSSIMLoss(kernel_size=5, kernel_sigma=1.5)
}


#  Try to make all metrics have scale ~10
METRIC_SCALE_FROM_NAME = {
    "ssim": 10.,
    "ms-ssim": 10.,
    "msid": 0.5,
    "fid": 0.1,
    "kid": 2.,
    "content": 3.,
    "style": 7e-8,
    "tv": 1.,
    "psnr": 0.3,
    "mse": 500.,
    "mae": 100.,
    "gmsd": 30.,
    "ms-gmsd": 50.,
    "vif": 30.,
    "gs": 3.,
    "loss": 1.  # not used
}
