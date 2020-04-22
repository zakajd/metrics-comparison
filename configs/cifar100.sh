#!/bin/bash

## Denoise Unet
echo """
name : denoise_cifar100_lightaug_UNet_L2
## General
seed : 1
device : 2
datasets : ['cifar100']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 2
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:2'}, 'style', {'device' : 'cuda:2'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Deblur Unet
echo """
name : deblur_cifar100_lightaug_UNet_L2
## General
seed : 1
device : 2
datasets : ['cifar100']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 2
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:2'}, 'style', {'device' : 'cuda:2'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Super Res Unet

## Denoise Unet
echo """
name : denoise_cifar100_lightaug_DnCNN_L2
## General
seed : 1
device : 2
datasets : ['cifar100']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 2
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:2'}, 'style', {'device' : 'cuda:2'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'dncnn'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml

## Deblur Unet
echo """
name : deblur_cifar100_lightaug_DnCNN_L2
## General
seed : 1
device : 2
datasets : ['cifar100']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 2
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:2'}, 'style', {'device' : 'cuda:2'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : 'dncnn'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 256

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/mnist.yaml


python3 train.py -c configs/mnist.yaml
