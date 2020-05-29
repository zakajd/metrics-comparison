#!/bin/bash
gpu="1"
task="deblur"

for model in "dncnn" # "unet"
do
    for dataset in "bsds100" "div2k" # "medicaldecathlon"
    do
        echo """
        name : ${task}_${model}_${dataset}_L1_MS-SSIM

        ## General
        device : 0
        train_dataset : $dataset
        val_datasets: [$dataset,'set5']
        task : $task
        check_val_every_n_epoch : 2
        metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'tv', {}, 'content', {}, 'style', {}]\"
        feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"

        ## Training parameters
        size : 256
        model : $model
        feature_extractor : 'inceptionV3'
        epochs : 60
        batch_size : 32

        lr : 1e-3
        weight-decay : 1e-4
        # model_params : {'num_layers' : 5}
        """ > configs/deblur.yaml
        CUDA_VISIBLE_DEVICES=${gpu} taskset -c 0-5 python3 train.py -c configs/deblur.yaml
    done
done
