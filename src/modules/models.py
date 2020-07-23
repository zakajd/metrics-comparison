# import random
# from functools import partial


import torch
import torch.nn as nn
# import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net
        Args:
            in_channels (int): Number of channels in input image
            out_channels (int): Number of channels in output image
            upsample (bool): Flag to increase image size 2 times for SR task"""

        super(UNet, self).__init__()

        # Upsamle
        upsample_module = nn.Upsample(scale_factor=2, mode='bilinear')

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            upsample_module)

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            upsample_module)

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            upsample_module)

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        return self._block6(concat1).tanh()


class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super().__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True)
                )
            )
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.layers(x)
        return (x - residual).tanh()


MODEL_FROM_NAME = {
    "unet": UNet,
    "dncnn": DnCNN,
}


class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Parameter(torch.ones(1) * 1e-3)
        self.b2 = nn.Parameter(torch.ones(1) * 1e-3)
        self.b3 = nn.Parameter(torch.ones(1) * 1e-3)
        self.b4 = nn.Parameter(torch.ones(1) * 1e-3)
        self.b5 = nn.Parameter(torch.zeros(1))

    def forward(self, predicted_scores):
        adjusted = self.b1 * (0.5 - 1 / (1 + torch.exp(self.b2 * (predicted_scores - self.b3)))) + \
            predicted_scores * self.b4 + self.b5
        return adjusted


class Discriminator(nn.Module):
    r"""Simple discriminator used in GAN training
    Args:
        nf (int): Number of discriminator features at the beginning
    Returns:
        logits: raw prediction for each image to be real or fake
    """
    def __init__(self, nf=32):
        super().__init__()

        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 2) x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 4) x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf * 8) x 4 x 4
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, input):
        return self.main(input).sigmoid()
