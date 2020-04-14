import random
from functools import partial


import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  mode="fan_out", nonlinearity="relu")
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


MODEL_FROM_NAME = {
    "unet": UNet,
    "dncnn": DnCNN,
}



class VGG(nn.Module):
    """
    Args:
        pretrained (str): if present, returns a model pre-trained on 'str' dataset
        num_classes (int, optional): [description]. Defaults to 1000.
        in_channels (int): Number of input (color) channels. Defaults to 3.
        norm_layer (ABN, optional): Which version of ABN to use. Choices are:
            'ABN' - dropin replacement for BN+Relu.
            'InplaceABN' - efficient version. If used with `pretrain` Weights still 
                will be loaded but performance may be worse than with ABN. 
    """

    def __init__(
        self,
        layers,
        num_classes=1000,
        in_channels=3,
        norm_layer="abn",
        norm_act="relu",
    ):

        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.norm_act = norm_act # !!!
        self.norm_layer = bn_from_name(norm_layer) # !!!
        
        self.features = self._make_layers(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._init_weights()


    def logits(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, self.norm_layer(v, activation=self.norm_act)]
                in_channels = v
        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        # filter classifier and num_batches_tracked
        for k in keys:
            if k.startswith("classifier") and self.encoder:
                state_dict.pop(k)
        # there is a mismatch in feature layers names, so need this mapping
        self_feature_names = [i for i in self.state_dict().keys() if "features" in i]
        load_feature_names = [i for i in state_dict.keys() if "features" in i]
        features_map = {load_f: self_f for (load_f, self_f) in zip(load_feature_names, self_feature_names)}
        for k in keys:
            if k.startswith("features"):
                state_dict[features_map[k]] = state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="linear")


CFGS = {
    "vgg16_bn": {
        "default": {
            "params": {
                "layers": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
            },
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        },
        "imagenet": {"url": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"},
    },
}


def _vgg(arch, pretrained="imagenet", **kwargs):
    cfgs = deepcopy(CFGS)
    cfg_settings = cfgs[arch]["default"]
    cfg_params = cfg_settings.pop("params")
    if pretrained:
        pretrained_settings = cfgs[arch][pretrained]
        pretrained_params = pretrained_settings.pop("params", {})
        cfg_settings.update(pretrained_settings)
        cfg_params.update(pretrained_params)
    common_args = set(cfg_params.keys()).intersection(set(kwargs.keys()))
    assert (
        common_args == set()
    ), "Args {} are going to be overwritten by default params for {} weights".format(
        common_args.keys(), pretrained or "default"
    )
    kwargs.update(cfg_params)
    model = VGG(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(cfgs[arch][pretrained]["url"])
        kwargs_cls = kwargs.get("num_classes", None)

            # if there is last_linear in state_dict, it's going to be overwritten
            state_dict["classifier.6.weight"] = model.state_dict()["classifier.6.weight"]
            state_dict["classifier.6.bias"] = model.state_dict()["classifier.6.bias"]
        model.load_state_dict(state_dict)
    setattr(model, "pretrained_settings", cfg_settings)
    return model


def vgg16_bn(**kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg("vgg16_bn", pretrained="imagenet", **kwargs)