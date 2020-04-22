#!/bin/bash

## Denoise Unet
echo """
name : denoise_TinyImegeNet_lightaug_UNet_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.001
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml


taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml

## Deblur Unet
echo """
name : deblur_TinyImegeNet_lightaug_UNet_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.001
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml


taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml

## Super Res Unet
echo """
name : sr_TinyImegeNet_lightaug_Unet_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'sr'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'unet'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.001
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml

taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml

## Denoise DnCNN
echo """
name : denoise_TinyImegeNet_lightaug_DnCNN_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'denoise'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'dncnn'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml

taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml

## Deblur DnCNN
echo """
name : deblur_TinyImegeNet_lightaug_DnCNN_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'deblur'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'dncnn'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.0003
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml

taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml

##  Super Res DnCNN
echo """
name : sr_TinyImegeNet_lightaug_DnCNN_L2
## General
seed : 1
device : 1
datasets : ['tinyimagenet']
data_mean : [0.0, 0.0, 0.0]
data_std : [1., 1., 1.]
aug_type : 'light'
task : 'sr'
compute_metrics_repeat : 3
check_val_every_n_epoch : 5
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {'device' : 'cuda:1'}, 'style', {'device' : 'cuda:1'}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : 'dncnn'
model_params: {}
feature_extractor : 'resnet18'
epochs : 60
batch_size : 128

lr : 0.001
momentum : 0.9
weight-decay : 0.0001
""" > configs/tinyimagenet.yaml

taskset -c 0-3 python3 train.py -c configs/tinyimagenet.yaml