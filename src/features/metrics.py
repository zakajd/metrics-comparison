import piq
import torch
import functools

from src.features.wrappers import BRISQUEWrapper, ISWrapper


# Init metrics with predefined parameters
METRIC_FROM_NAME = {
    # Full Reference
    "mae": torch.nn.L1Loss(),
    "mse": torch.nn.MSELoss(),
    "psnr": piq.psnr,
    "psnr_y": functools.partial(
        piq.psnr, convert_to_greyscale=True),
    "ssim": functools.partial(
        piq.ssim, kernel_size=5, kernel_sigma=1.5),
    "ms-ssim": functools.partial(
        piq.multi_scale_ssim, kernel_size=5, kernel_sigma=1.5),
    "vifp": functools.partial(
        piq.vif_p, sigma_n_sq=1.0),
    "vifp_2": functools.partial(
        piq.vif_p, sigma_n_sq=2.0),
    "gmsd": piq.gmsd,
    "ms-gmsd": piq.multi_scale_gmsd,
    "ms-gmsdc": functools.partial(
        piq.multi_scale_gmsd, chromatic=True),
    "fsim": functools.partial(
        piq.fsim, chromatic=False),
    "fsimc": functools.partial(
        piq.fsim, chromatic=True),
    "vsi": piq.vsi,
    "mdsi": piq.mdsi,

    "content_vgg16": piq.ContentLoss(
        layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=True),
    "content_vgg16_ap": piq.ContentLoss(
        layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=True,
        replace_pooling=True),

    "content_vgg19": piq.ContentLoss(
        feature_extractor='vgg19',
        layers=['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=True),
    "content_vgg19_ap": piq.ContentLoss(
        feature_extractor='vgg19',
        layers=['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=True,
        replace_pooling=True),

    "style_vgg16": piq.StyleLoss(
        feature_extractor='vgg16',
        layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=False),
    "style_vgg19": piq.StyleLoss(
        feature_extractor='vgg19',
        layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
        normalize_features=False),

    "lpips": piq.LPIPS(replace_pooling=False),
    "lpips_ap": piq.LPIPS(replace_pooling=True),
    "dists": piq.DISTS(),

    # No reference
    "brisque": BRISQUEWrapper(),

    # Distribution based metrics
    "fid": piq.FID(),
    "kid": piq.KID(),
    "gs": piq.GS(sample_size=64, num_iters=500, num_workers=4, i_max=256),
    "is_metric": ISWrapper(num_splits=3),
    "is": piq.IS(num_splits=3),
    "msid": piq.MSID(),
}


def pearson_correlation(x, y, invert=False):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    if invert:
        return 1 - corr
    return corr
