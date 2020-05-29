#!/bin/bash
gpu="1"
task="sr"

for model in "dncnn" "unet"
do
    for dataset in "bsds100" "div2k" # "medicaldecathlon"
    do
        echo """
        name : ${task}_${model}_${dataset}_L1_MS-SSIM

        ## General
        device : 0
        train_dataset : $dataset
        val_datasets: [$dataset,'set5']
        aug_type : 'medium'
        task : $task
        check_val_every_n_epoch : 2
        metrics: \"['ssim', {}, 'ms-ssim', {}, 'psnr', {}, 'content', {}, 'style', {}, 
                    'mse', {}, 'mae', {}, 'gmsd', {}, 'ms-gmsd', {}, 'vif', {}]\"
        feature_metrics: \"['kid', {}, 'fid', {}, 'msid', {}]\"
        #  'gs', {} NOT INCLUDED YET 

        ## Training parameters
        size : 256
        model : $model
        feature_extractor : 'inceptionV3'
        epochs : 60
        batch_size : 16

        lr : 3e-4
        weight-decay : 1e-4
        # model_params : {'num_layers' : 5}
        """ > configs/sr.yaml
        CUDA_VISIBLE_DEVICES=${gpu} taskset -c 10-15 python3 train.py -c configs/sr.yaml
    done
done
