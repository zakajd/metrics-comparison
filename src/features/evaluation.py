"""Function to compute metrics on TID2013 and KADID10k datasets.
Function to compute PLCC, SRCC, KRCC scores
"""
import os
import functools
import collections
from typing import List

import piq
import torch
import numpy as np
import torchvision
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau

from src.modules.models import Regression
from src.data.utils import crop_patches
from src.data.datasets import TID2013, KADID10k, DistortionSampler

# Useful for some metrics
torch.multiprocessing.set_sharing_strategy('file_system')
DATASET_FROM_NAME = {
    "tid2013": TID2013,
    "kadid10k": KADID10k,
}

# Init metrics

METRIC_FROM_NAME = {
    # Full Reference
    "psnr": functools.partial(piq.psnr, reduction='none', data_range=1.0),
    "ssim": functools.partial(piq.ssim, data_range=1.0, size_average=False, full=False),
    "ms_ssim": functools.partial(piq.multi_scale_ssim, data_range=1.0, size_average=False),
    "vif_p": functools.partial(piq.vif_p, sigma_n_sq=1.0, data_range=1.0, reduction='none'),
    "vif_p_2": functools.partial(piq.vif_p, sigma_n_sq=2.0, data_range=1.0, reduction='none'),
    "gmsd": piq.GMSDLoss(reduction='none', data_range=1.0),
    "ms_gmsd": piq.MultiScaleGMSDLoss(reduction='none', data_range=1.0),
    "ms_gmsdc": piq.MultiScaleGMSDLoss(chromatic=True, reduction='none', data_range=1.0),
    "fsim": functools.partial(piq.fsim, data_range=1.0, chromatic=False, reduction='none'),
    "fsimc": functools.partial(piq.fsim, data_range=1.0, chromatic=True, reduction='none'),
    "vsi": functools.partial(piq.vsi, data_range=1.0, reduction='none'),
    "mdsi": functools.partial(piq.mdsi, data_range=1.0, reduction='none'),

    # "content_vgg16": piq.ContentLoss(
    #     layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=True,
    #     reduction='none'),
    # "content_vgg16_ap": piq.ContentLoss(
    #     layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=True,
    #     replace_pooling=True,
    #     reduction='none'),

    # "content_vgg19": piq.ContentLoss(
    #     feature_extractor='vgg19',
    #     layers=['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=True,
    #     reduction='none'),
    # "content_vgg19_ap": piq.ContentLoss(
    #     feature_extractor='vgg19',
    #     layers=['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=True,
    #     replace_pooling=True,
    #     reduction='none'),

    # "style_vgg16": piq.StyleLoss(
    #     feature_extractor='vgg16',
    #     layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=False,
    #     reduction='none'),
    # "style_vgg19": piq.StyleLoss(
    #     feature_extractor='vgg19',
    #     layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3'],
    #     weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    #     normalize_features=False,
    #     reduction='none'),

    "lpips_ap": piq.LPIPS(replace_pooling=True, reduction="none"),
    "dists": piq.DISTS(reduction="none"),

    # No reference
    "brisque": functools.partial(piq.brisque, data_range=1.0, reduction='none'),

    # Distribution based metrics
    "fid": piq.FID(),
    "kid": piq.KID(),
    "gs": piq.GS(sample_size=64, num_iters=500, num_workers=4, i_max=256),
    "is": piq.IS(num_splits=3),
    "msid": piq.MSID(),
}


def compute_metrics(
        dataset_name: str,
        transform,
        batch_size: int,
        metrics_list: List[str],
        extractor_name: str,
        fullref: bool = False,
        noref: bool = False,
        dist: bool = False):
    """Compute metrics on a given dataset
    Args:
        name: Dataset name. One of {'tid2013', 'koniq10k'}
        transform: albumentations.transform
        batch_size: Number of images to sample at once
        metrics_list: Metrics to compute.
        feature_extractor: Model used to extract image features
        noref: Flag to compute BRISQUE
        dist: Flag to compute distribution based metrics
    Returns:
        metric_scores: Dict with keys same as metric list and values torch.Tensors of results
    """
    distortions = DATASET_FROM_NAME[dataset_name]().df['dist_type'].unique()
    metric_scores = collections.defaultdict(list)

    # Init metrics
    metrics = [METRIC_FROM_NAME[metric] for metric in metrics_list]

    if dist and (extractor_name == "vgg16"):
        feature_extractor = torchvision.models.vgg16(pretrained=True, progress=True).features.to("cuda")
    elif dist and (extractor_name == "vgg19"):
        feature_extractor = torchvision.models.vgg19(pretrained=True, progress=True).features.to("cuda")
    elif dist and (extractor_name == "inception"):
        feature_extractor = piq.feature_extractors.InceptionV3(
            resize_input=False, use_fid_inception=True, normalize_input=True).to("cuda")
    elif dist:
        raise ValueError("Wrong feature extractor name")

    # Iterate over distortions:
    for distortion in tqdm(distortions):
        # Reinit dataset
        dataset = DATASET_FROM_NAME[dataset_name](transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=DistortionSampler(dataset, dist_type=distortion),
            drop_last=False,
        )

        distorted_features, reference_features = [], []

        for (distorted_images, reference_images, score) in loader:
            distorted_images, reference_images = distorted_images.to("cuda"), reference_images.to("cuda")
            # metric_scores['score'].append(score.cpu().numpy())

            if fullref:
                for name, metric in zip(metrics_list, metrics):
                    metric_scores[name].extend(metric(distorted_images, reference_images).cpu().numpy())

            if noref:
                for name, metric in zip(metrics_list, metrics):
                    metric_scores[name].extend(metric(distorted_images).cpu().numpy())

            if dist:
                # Create patches
                distorted_patches = crop_patches(distorted_images, size=96, stride=32)
                reference_patches = crop_patches(reference_images, size=96, stride=32)

                # Extract features from distorted images
                patch_loader = distorted_patches.view(-1, 10, *distorted_patches.shape[-3:])
                for patches in patch_loader:
                    with torch.no_grad():
                        if extractor_name == "inception":
                            features = feature_extractor(patches)
                            distorted_features.append(features[0].squeeze())
                        elif extractor_name == "vgg16":
                            features = torch.nn.functional.avg_pool2d(feature_extractor(patches), 3)
                            distorted_features.append(features.squeeze())
                        elif extractor_name == "vgg19":
                            features = torch.nn.functional.avg_pool2d(feature_extractor(patches), 3)
                            distorted_features.append(features.squeeze())

                # Extract features from reference images
                patch_loader = reference_patches.view(-1, 10, *reference_patches.shape[-3:])
                for patches in patch_loader:
                    with torch.no_grad():
                        if extractor_name == "inception":
                            features = feature_extractor(patches)
                            reference_features.append(features[0].squeeze())
                        elif extractor_name == "vgg16":
                            features = torch.nn.functional.avg_pool2d(feature_extractor(patches), 3)
                            reference_features.append(features.squeeze())
                        elif extractor_name == "vgg19":
                            features = torch.nn.functional.avg_pool2d(feature_extractor(patches), 3)
                            reference_features.append(features.squeeze())
        if dist:
            distorted_features = torch.cat(distorted_features, dim=0)
            reference_features = torch.cat(reference_features, dim=0)

            for name, metric in zip(metrics_list, metrics):
                score = metric(distorted_features, reference_features)
                metric_scores[name].extend([score.cpu().numpy()] * len(dataset))

    return metric_scores


def save_scores(dataset_name, metric_scores):
    if not os.path.exists(f'data/interim/{dataset_name}'):
        os.mknod(f'data/interim/{dataset_name}')

    for key in metric_scores.keys():
        value = metric_scores[key]

        # Reduce
        scores = np.stack(value)

        # Delete old file and create new one
        try:
            os.remove(f"data/interim/{dataset_name}/{key}.txt")
        except OSError:
            pass

        with open(f"data/interim/{dataset_name}/{key}.txt", "w") as file:
            # file.write('\n'.join(str(score) for score in scores.cpu().numpy()))
            file.write('\n'.join(str(score) for score in scores))
        print(key, len(scores))


def fit_regression(
        dataset_name, metrics_list, lr: float = 1e-4, epochs: int = 10000, clip_value: float = 1., num_folds=1):
    """Train regresion on scores"""

    # Read true scores
    with open(f"data/interim/{dataset_name}/score.txt") as f:
        mos_scores = f.readlines()
    mos_scores = [float(score) for score in mos_scores]

    metric_results = {metric: collections.defaultdict(list) for metric in metrics_list}
    for metric in metrics_list:

        # Read metric scores
        with open(f"data/interim/{dataset_name}/{metric}.txt") as file:
            scores = file.readlines()

        scores = [float(score) for score in scores]
        indexes = list(range(len(scores)))
        np.random.shuffle(indexes)

        # K-fold validation of results
        fold_size = len(scores) // num_folds
        for k in range(num_folds):
            idx = indexes[k * fold_size: (k + 1) * fold_size]

            model = Regression()
            criterion = torch.nn.MSELoss()

            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.05)

            prediction = torch.tensor(scores)[idx] * (1e-10 if 'style' in metric else 1)
            target = torch.tensor(mos_scores)[idx]

            # Fit regression
            for epoch in range(epochs + 1):
                # Clear gradient
                optimizer.zero_grad()

                # get output from the model, given the inputs
                outputs = model(prediction)

                # Get loss for the predicted output
                loss = criterion(outputs, target)
                if torch.isnan(loss):
                    print("NaN in loss!")
                    continue

                # get gradients w.r.t to parameters
                loss.backward()

                # clip grads
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                # update parameters
                optimizer.step()
                scheduler.step(loss)

            # Compute metrics:
            x = outputs.detach().numpy()
            y = target.detach().numpy()
            metric_results[metric]['PLCC'].append(pearsonr(x, y)[0])
            metric_results[metric]['SRCC'].append(spearmanr(x, y)[0])
            metric_results[metric]['KRCC'].append(kendalltau(x, y)[0])

        print(f"{metric}:",
              f"PLCC {np.mean(metric_results[metric]['PLCC']):0.3f} \u00B1 {np.std(metric_results[metric]['PLCC']):0.3f}",
              f"SRCC {np.mean(metric_results[metric]['SRCC']):0.3f} \u00B1 {np.std(metric_results[metric]['SRCC']):0.3f}",
              f"KRCC {np.mean(metric_results[metric]['KRCC']):0.3f} \u00B1 {np.std(metric_results[metric]['KRCC']):0.3f}")
