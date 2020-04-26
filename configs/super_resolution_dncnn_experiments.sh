#!/bin/bash
#!/bin/bash
gpu="1"
model="dncnn"

# "set5"
# "set14"
# "urban100":
# "manga109"

#------------
task="sr"
dataset="mnist"

echo """
name : ${task}_${model}_${dataset}_L2
## General
device : 0
datasets : [$dataset]
task : $task
check_val_every_n_epoch : 3
num_images_to_log: 4
metrics: \"['ssim', {'kernel_size' : 3}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
## Training parameters
size : 32
model : $model
feature_extractor : 'resnet18'
epochs : 60
batch_size : 512
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml

#------------
task="sr"
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
""" > configs/sr.yaml
CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/sr.yaml
