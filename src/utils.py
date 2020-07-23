import piq
import torch
import torchvision
from src.modules.wrappers import InceptionV3Wrapper, BRISQUEWrapper, ISWrapper
from src.modules.losses import PSNR

EXTRACTOR_FROM_NAME = {
    "vgg16": torchvision.models.vgg16(pretrained=True, progress=False).features,
    "vgg19": torchvision.models.vgg19(pretrained=True, progress=False).features,
    "inception": InceptionV3Wrapper(resize_input=False, use_fid_inception=True, normalize_input=True),
}


METRIC_FROM_NAME = {
    # FR
    "mae": torch.nn.L1Loss,
    "mse": torch.nn.MSELoss,
    "psnr": PSNR,
    "ssim": piq.SSIMLoss,
    "ms-ssim": piq.MultiScaleSSIMLoss,
    # "iw-ssim": piq.iw_ssim,
    "gmsd": piq.GMSDLoss,
    "ms-gmsd": piq.MultiScaleGMSDLoss,
    "ms-gmsdc": piq.MultiScaleGMSDLoss,
    "fsim": piq.FSIMLoss,
    "fsimc": piq.FSIMLoss,
    "vsi": piq.VSILoss,
    "mdsi": piq.MDSILoss,
    "vif": piq.VIFLoss,

    "content": piq.ContentLoss,
    "content_ap": piq.ContentLoss,
    "style": piq.StyleLoss,
    "style_ap": piq.StyleLoss,
    "lpips": piq.LPIPS,
    "dists": piq.DISTS,

    # NR
    "brisque": BRISQUEWrapper,

    # Distrib based
    "is_metric": ISWrapper,
    "is": piq.IS,
    "fid": piq.FID,
    "gs": piq.GS,
    "kid": piq.KID,
    "msid": piq.MSID,
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
