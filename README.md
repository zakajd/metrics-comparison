# Repository for the experimental metrics comparison

Available datasets:
- MNIST
- FashionMNIST
- CIFAR10
- CIFAR100
- TinyImageNet

Task:
- Denoising
- Deblurring

Metrics:
- PSNR, SSIM, MSID, FID, KID, Total variation, ...

## Installation
- If you use environment managing systems, create a new one `conda env create -n "metrics"`
- Install all requiered packages `pip3 install -r requirements.txt`

## Code organization

    ├── README.md             <- Top-level README.
    |
    ├── confings    <- Parameters for traing 
    │   ├── base.yaml         <- Default parameters
    │
    ├── datasets        <- All data must be here
    |
    ├── docker        <- Container for easy reprodusable solution (NOT working yet)
    ├── logs        <- TensorBoard logging to monitor training
    ├── models        <- Pretrained models saved as `*.ckpt`
    ├── notebooks        <- Jupyter Notebooks
    │   ├── Testing.ipynb   <- Develompent related stuff                  
    │   ├── Demo.ipynb  <- Demonstation of the results
    ├── src        <- Code

## Loading data
Run `bash data_download.sh` script. Can be VERY slow due to lowbandwidth of datahosting servers. Alternatively load datasets from Kaggle one by one. 


## Training Example
Launch docker container (not available for now)

- Edit `configs/base.yaml` to match your system capacity (batch size, # of GPUs, etc.)
- Run `python3 train.py --config_file configs/base.yaml`

Logs are writen into `logs/` dir by default.
