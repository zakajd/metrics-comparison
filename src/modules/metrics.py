from typing import List, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19

from piq.utils import _validate_input, _adjust_dimensions
from piq.functional import similarity_map

# Map VGG names to corresponding number in torchvision layer
VGG16_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "pool3": '16',
    "conv4_1": '17', "relu4_1": '18',
    "conv4_2": '19', "relu4_2": '20',
    "conv4_3": '21', "relu4_3": '22',
    "pool4": '23',
    "conv5_1": '24', "relu5_1": '25',
    "conv5_2": '26', "relu5_2": '27',
    "conv5_3": '28', "relu5_3": '29',
    "pool5": '30',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pearson_correlation(x, y, invert=False):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    if invert: 
        return 1 - corr
    return corr


class DISTS(_Loss):
    r"""Deep Image Structure and Texture Similarity metric.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]

    References:
        .. [1] Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli
        (2020). Image Quality Assessment: Unifying Structure and Texture Similarity.
        https://arxiv.org/abs/2004.07728
    """
    # Constant used in feature normalization to avoid zero division
    EPS = 1e-10
    _weights_url = "https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt"

    def __init__(self, weights: List[Union[float, torch.Tensor]] = [1.], 
                 reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD) -> None:
        r"""
        Args:
            layers: List of strings with layer names. Default: [`relu3_3`]
            reduction: Reduction over samples in batch: "mean"|"sum"|"none"
            mean: List of float values used for data standartization. Default: ImageNet mean.
                If there is no need to normalize data, use [0., 0., 0.].
            std: List of float values used for data standartization. Default: ImageNet std.
                If there is no need to normalize data, use [1., 1., 1.].

        """
        super().__init__()

        layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]
        self.model = vgg16(pretrained=True, progress=False).features
        self.layers = [VGG16_LAYERS[l] for l in layers]

        # Replace MaxPooling layer with L2Pooling. See [1] for details.
        self.model = self.max_pool_to_l2_pool(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        # self.distance = ...

        dists_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        self.alpha_weights = torch.split(dists_weights['alpha'], channels, dim=1)
        self.beta_weights = torch.split(dists_weights['beta'], channels, dim=1)

        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of DISTS loss between feature representations of prediction and target tensors.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        # Normalize input
        mean, std = self.mean.to(prediction), self.std.to(prediction)
        prediction = (prediction - mean) / std
        target = (target - mean) / std

        self.model.to(prediction)
        prediction_features = self.get_features(prediction)
        prediction_features.insert(0, prediction)
        target_features = self.get_features(target)
        target_features.insert(0, target)
        
        distances = self.compute_distance(prediction_features, target_features)

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        structure_loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances[0], self.alpha_weights)], dim=1).sum(dim=1)
        texture_loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances[1], self.beta_weights)], dim=1).sum(dim=1)

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def compute_distance(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute structure similarity feature maps"""
        distances = [[], []]

        for x, y in zip(prediction_features, target_features):
            x_mean = x.mean([2, 3], keepdim=True)
            y_mean = x.mean([2 ,3], keepdim=True)
            distances[0].append(similarity_map(x_mean, y_mean, constant=1e-6))

            x_var = ((x - x_mean) ** 2).mean([2,3], keepdim=True)
            y_var = ((y - y_mean) ** 2).mean([2,3], keepdim=True)
            xy_cov = (x * y).mean([2,3], keepdim=True) - x_mean * y_mean
            distances[1].append((2 * xy_cov + 1e-6) / (x_var + y_var + 1e-6))

        return distances

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: torch.Tensor with shape (N, C, H, W)
        
        Returns:
            features: List of features extracted from intermediate layers
        """
        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features

    def max_pool_to_l2_pool(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into L2Pool"""
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = L2Pool2d(kernel_size=3, stride=2, padding=1)
            
        for name, child in module.named_children():
            module_output.add_module(name, self.max_pool_to_l2_pool(child))
        return module_output


class L2Pool2d(torch.nn.Module):
    r"""Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12
    def __init__(self, kernel_size: int = 3, stride: int = 2, padding=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = None

    def forward(self, x):
        if self.kernel is None:
            self.construct_kernel(x)
    
        out = nn.functional.conv2d(x ** 2, self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + self.EPS).sqrt()
    
    def construct_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 2D Hann window kernel with number of channels equal to input channels"""
        C = x.size(1)
        
        # Take bigger window and drop borders
        window = torch.hann_window(self.kernel_size + 2, periodic=False)[1:-1]
        kernel = window[:,None] * window[None,:]

        # Normalize and reshape kernel
        self.kernel = (kernel / kernel.sum()).repeat((C,1,1,1))