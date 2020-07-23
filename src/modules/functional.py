from typing import List
import collections

import torchvision
from piq.perceptual import VGG16_LAYERS, VGG19_LAYERS

from src.utils import METRIC_FROM_NAME


def listify(p):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    return p


def init_metrics(metrics: List, model_name: str = "vgg16"):
    """Initialize metrics and use single feature extractor for all VGG based metrics
    Args:
        metrics: List in format [`metric_name`, {`metric_params`}, ... ]
        model_name: One of {`vgg16`, `vgg19`}
    """
    model = {
        "vgg16": torchvision.models.vgg16(pretrained=True, progress=False).features,
        "vgg19": torchvision.models.vgg19(pretrained=True, progress=False).features,
    }

    layers_dict = {
        "vgg16": VGG16_LAYERS,
        "vgg19": VGG19_LAYERS,
    }
    metric_names = metrics[::2]
    result = []
    for name, kwargs in zip(metric_names, metrics[1::2]):
        # Init with fixed feature encoder instead of creating new one inside metric
        if ("style" in name) or ("content" in name):
            feature_extractor = kwargs.pop("feature_extractor")
            layers = [layers_dict[feature_extractor][l] for l in kwargs.pop("layers")]
            metric = METRIC_FROM_NAME[name](
                feature_extractor=model[feature_extractor],
                layers=layers,
                **kwargs)
            metric.name = f"{name}_{feature_extractor}"
            result.append(metric)
        else:
            metric = METRIC_FROM_NAME[name](**kwargs)
            metric.name = name
            result.append(metric)
    return result
