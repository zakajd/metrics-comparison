import os
import random
from typing import Tuple, Union

import torch
from torch.nn.modules.loss import _Loss
import numpy as np
import photosynthesis_metrics as pm


from src.losses import StyleLoss, ContentLoss, PSNR


class SumOfLosses(_Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)


class WeightedLoss(_Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = torch.Tensor([weight])

    def forward(self, *inputs):
        loss = self.loss(*inputs)
        self.weight = self.weight.to(loss.device)
        return loss * self.weight[0]

EXTRACTOR_FROM_NAME = {
    "vgg16": torchvision.models.vgg16(pretrained=True, progress=False).features,
    "vgg19": torchvision.models.vgg19(pretrained=True, progress=False).features,
    "inception": InceptionV3(resize_input=False, use_fid_inception=True, normalize_input=True),
}

METRIC_FROM_NAME = {
    "ssim": pm.SSIMLoss,
    "ms-ssim": pm.MultiScaleSSIMLoss,
    "msid": pm.MSID,
    "fid": pm.FID,
    "kid": pm.KID,
    "content": ContentLoss,
    "style": StyleLoss,
    "tv": pm.TVLoss,
    "psnr": PSNR,
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
    "gmsd": pm.GMSDLoss,
    "ms-gmsd": pm.MultiScaleGMSDLoss,
    "vif": pm.VIFLoss,
    "gs": pm.GS,
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
    "gs":3.,
    "loss": 1.  # not used
}
