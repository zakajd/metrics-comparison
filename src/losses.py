"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
import collections


import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss 
from torchvision.models import vgg16



def listify(p):
    if p is None:
        p = []
    elif not isinstance(p, collections.Iterable):
        p = [p]
    return p

class PSNR(_Loss):
    def forward(self, prediction, target):
        mse = torch.mean((prediction - target) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr



class ContentLoss(_Loss):
    """
    Creates content loss for neural style transfer.
    Uses pretrained VGG16 model from torchvision by default
    layers: list of VGG layers used to evaluate content loss
    criterion: str in ['mse', 'mae'], reduction method
    """

    def __init__(
        self,
        layers=["21"],
        weights=1,
        loss="mse",
        device="cuda",
        **args,
    ):
        super().__init__()
        self.model = vgg16(pretrained=True, **args)
        self.model.eval().to(device)
        self.layers = listify(layers)
        self.weights = listify(weights)

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, content):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = torch.stack(self.get_features(input))
        content_features = torch.stack(self.get_features(content))
        loss = self.criterion(input_features, content_features)

        # Solve big memory consumption
        torch.cuda.empty_cache()
        return loss

    def get_features(self, x):
        """
        Extract feature maps from the intermediate layers.
        """
        if self.layers is None:
            self.layers = ["21"]

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        # print(len(features))
        return features


class StyleLoss(_Loss):
    """
    Class for creating style loss for neural style transfer
    model: str in ['vgg16_bn']
    """

    def __init__(
        self,
        layers=["0", "5", "10", "19", "28"],
        weights=[0.75, 0.5, 0.2, 0.2, 0.2],
        loss="mse",
        device="cuda",
        **args,
    ):
        super().__init__() 
        self.model = vgg16(pretrained=True, **args)
        self.model.eval().to(device)

        self.layers = listify(layers)
        self.weights = listify(weights)

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise KeyError

    def forward(self, input, style):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = self.get_features(input)
        style_features = self.get_features(style)
        # print(style_features[0].size(), len(style_features))

        input_gram = [self.gram_matrix(x) for x in input_features]
        style_gram = [self.gram_matrix(x) for x in style_features]

        loss = [
            self.criterion(torch.stack(i_g), torch.stack(s_g)) for i_g, s_g in zip(input_gram, style_gram)
        ]
        return torch.mean(torch.tensor(loss))

    def get_features(self, x):
        """
        Extract feature maps from the intermediate layers.
        """
        if self.layers is None:
            self.layers = ["0", "5", "10", "19", "28"]

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features

    def gram_matrix(self, input):
        """
        Compute Gram matrix for each image in batch
        input: Tensor of shape BxCxHxW
            B: batch size
            C: channels size
            H&W: spatial size
        """

        B, C, H, W = input.size()
        gram = []
        for i in range(B):
            x = input[i].view(C, H * W)
            gram.append(torch.mm(x, x.t()))
        return gram