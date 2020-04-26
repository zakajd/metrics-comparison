#!/bin/bash
gpu="2"
model="unet"

# "set5"
# "set14"
# "urban100":
# "manga109"
#------------
task="denoise"
dataset="mnist"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 512
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="fashion_mnist"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 512
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="cifar10"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 512
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="cifar100"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 16
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 512
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="tinyimagenet"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'ms-ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 64
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 256
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="div2k"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 256
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 32
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="bsds100"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 256
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 32
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="coil100"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 256
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 32
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
#------------
task="denoise"
dataset="medicaldecathlon"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 256
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 32
""" > configs/denoise.yaml

CUDA_VISIBLE_DEVICES=${gpu} taskset -c 6-10 python3 train.py -c configs/denoise.yaml
