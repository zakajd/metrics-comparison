#!/bin/bash

## Denoise
echo """
name : denoise_TinyImegeNet_lightaug_UNet_L2
## General
seed : 1
device : 3
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 1
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 40
batch_size : 128

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Deblur
echo """
name : deblur_TinyImegeNet_lightaug_UNet_L2
## General
seed : 1
device : 3
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 1
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 40
batch_size : 128

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Super Res 