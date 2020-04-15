#!/bin/bash

## Denoise
echo """
name : denoise_cifar10_lightaug_UNet_L2
## General
seed : 1
device : 1
datasets : ['cifar10']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 1
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 40
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Deblur
echo """
name : deblur_cifar10_lightaug_UNet_L2
## General
seed : 1
device : 1
datasets : ['mnist']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 1
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 40
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Super Res 
# echo """
# name : sr_mnist_lightaug_UNet_L2
# ## General
# seed : 1
# device : 0
# datasets : ['mnist']
# data_mean : [0.0, 0.0, 0.0]
# data_std : [1., 1., 1.]
# aug_type : 'light'
# task : 'sr'
# compute_metrics_repeat : 3
# check_val_every_n_epoch : 1
# num_images_to_log: 16
# metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
# feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
# ## Training parameters
# size : 32
# model : 'unet'
# model_params: {}
# feature_extractor : 'resnet18'
# epochs : 40
# batch_size : 512

# lr : 0.0003
# momentum : 0.9
# weight-decay : 0.0001
# """ > configs/mnist.yaml


# python3 train.py -c configs/mnist.yaml
