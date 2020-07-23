"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
from typing import List
import functools

import piq
import torch
import torch.nn as nn


class SumOfLosses(torch.nn.modules.loss._Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)


class WeightedLoss(torch.nn.modules.loss._Loss):
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


class PSNR(nn.Module):
    def __init__(self, data_range=1.0, reduction='mean', convert_to_greyscale: bool = False):
        self.metric = functools.partial(
            piq.psnr, data_range=data_range, reduction=reduction, convert_to_greyscale=convert_to_greyscale)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        self.metric(prediction, target)


class GeneratorWGAN(nn.Module):
    r"""Compute loss for Generator model using equation for WGAN-GP
    Add additional regularization by adding MSE loss
    Args:
        weights: 2 float numbers, weight of MSE and Adversarial components
    """
    def __init__(self, weights: List[float] = [1.0, 1e-4]):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.weights = weights

    def forward(
            self, output: torch.Tensor, target: torch.Tensor, fake_logits: torch.Tensor, real_logits: torch.Tensor):
        r"""
        Args:
            output: Model prediciton for given input
            target: Ground truth image
            fake_logits: Discriminator predictions for output images
            real_logits: Discriminator predictions for real images
        """
        # L2 loss
        mse = self.criterion(output, target)

        # Adversarial loss: max D(G(z)) == min - D(G(z))
        adversarial_loss = - fake_logits.sigmoid().mean()

        loss = mse * self.weights[0] + adversarial_loss * self.weights[1]
        return loss


class DiscriminatorWGAN(nn.Module):
    """Compute loss for Discriminator model using equation for WGAN-GP

    Args:
        gp (bool): Use Gradient Penalty
        interpolate (bool): Interpolates between target and output images before GP
    """
    def __init__(self, gp=True, interpolate=False):
        super().__init__()
        self.gp = gp
        self.interpolate = interpolate

    def forward(
            self, output: torch.Tensor, target: torch.Tensor, fake_logits: torch.Tensor, real_logits: torch.Tensor):
        r"""
        Args:
            output: Model prediciton for given input
            target: Ground truth image
            fake_logits: Discriminator predictions for output images
            real_logits: Discriminator predictions for real images
        """
        # Adversarial loss: maximize D(x) - D(G(z))
        adversarial_loss = - (real_logits.sigmoid().mean() - fake_logits.sigmoid().mean())
        # # Gradient Penalty: (||grad D(x)|| - 1)^2
        # if self.gp:
        #     gradient_penalty = self._gradient_penalty(model_disc, target, output)
        #     adversarial_loss = adversarial_loss + gradient_penalty
        return adversarial_loss

    # def _gradient_penalty(self, model_disc, target, output, interpolate=False):
    #     if self.interpolate:
    #         # Calculate interpolation
    #         alpha = torch.rand(target.size(0), 1, 1, 1, device=target.device)
    #         interpolated = alpha * target + (1 - alpha) * output
    #         interpolated = interpolated.requires_grad_(True)
    #     else:
    #         interpolated = target.requires_grad_(True)

    #     # Always compute grads. in this part to avoid errors
    #     with torch.set_grad_enabled(True):
    #         # Calculate probability of interpolated examples
    #         prob_interpolated = model_disc(interpolated)#.requires_grad_(True)

    #         # Calculate gradients of probabilities with respect to examples
    #         gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
    #                                grad_outputs=torch.ones_like(prob_interpolated),
    #                                create_graph=True, retain_graph=True, allow_unused=True)[0]

    #     # Gradients have shape (batch_size, num_channels, img_width, img_height),
    #     # so flatten to easily take norm per example in batch
    #     gradients = gradients.view(target.size(0), -1)

    #     # Derivatives of the gradient close to 0 can cause problems because of
    #     # the square root, so manually calculate norm and add epsilon
    #     gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    #     # Return gradient penalty
    #     return 10 * ((gradients_norm - 1) ** 2).mean()
