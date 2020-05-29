# Defines functions to parse arguments from yaml files, instead of using command line arguments.
import os
import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="metrics-comparison",
        default_config_files=["configs/base.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument

    add_arg("--name", type=str, default="test", help="Folder name for logs")

    # General
    add_arg("--seed", type=int, default=42, help="Random seed for reproducable results")
    add_arg("--device", type=str, default="0", help="Device used for training. Can be `1, 2` or `-1`.")
    add_arg("--train_dataset", default=["cifar10"], type=str, nargs="+",
            help="Dataset to use for training. Default is CIFAR10")
    add_arg("--val_datasets", default=["set5"], type=str, nargs="+",
            help="Datasets to use for validation. Default is only Set5")
#     add_arg("--mean", type=float, nargs="+", default=[0.5, 0.5, 0.5],
#             help="Mean used to normalize data into [-1, 1] or N(0, 1)")
#     add_arg("--std", type=float, nargs="+", default=[1., 1., 1.],
#             help="STD used to normalize data into [-1, 1] or N(0, 1)")
    add_arg("--aug_type", type=str, default="light")
    add_arg("--task", type=str, default="denoise", choices=["denoise", "deblur", "sr"])
    add_arg("--data_mean", type=float, default=[0.5, 0.5, 0.5], nargs=3,
            help="Mean used for normalization")
    add_arg("--data_std", type=float, default=[0.5, 0.5, 0.5], nargs=3,
            help="Std used for normalization")
    add_arg("--feature_extractor", type=str, default="resnet18",
            help="Converts images into low-dimensional representation")
    add_arg("--check_val_every_n_epoch", type=int, default=5,
            help="How often to run validation step")
    add_arg("--compute_metrics_repeat", type=int, default=1)
    add_arg("--num_images_to_log", type=int, default=4, help="How many images to plot each val epoch")

    #  Training parameters
    add_arg('--model', type=str, default='unet', help='Model name')
    add_arg("--model_params", type=eval, help="Additional model params as kwargs")
#     add_arg('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    add_arg('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    add_arg("--batch_size", type=int, default=32, help="Number of images in stack")
    add_arg("--size", type=int, default=32, help="Size of image crop")
    add_arg("--lr", type=float, default=0.0001, help="Learning rate")
    add_arg('--momentum', default=0.9, type=float, metavar='M',
            help='momentum')
    add_arg('--wd', '--weight-decay', default=1e-4, type=float,
            metavar='W', help='weight decay', dest='weight_decay')
    add_arg("--metrics", type=eval, help="Image metrics and their parameters")
    add_arg("--feature_metrics", type=eval, help="Feature metrics and their parameters")

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    # If folder already exist append version number
    outdir = os.path.join("logs/", args.name)
    if os.path.exists(outdir):
        version = 1
        while os.path.exists(outdir):
            outdir = os.path.join("logs/", args.name + "_" + str(version))
            version += 1

    args.outdir = outdir
    return args
