# Defines functions to parse arguments from yaml files, instead of using command line arguments. 
import os
import configargparse as argparse
# from pytorch_tools.utils.misc import get_timestamp


def get_parser():
    parser = argparse.ArgumentParser(
        description="voice-transfer",
        default_config_files=["configs/base.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument

    ## Data
    # add_arg("--data_path", type=str, help="Path to raw data")
    add_arg("--name", type=str, default="arctic", help="Name of the dataset to use")
    add_arg("--extension", type=str, default=".wav", help="Files extencion")


    ## Training parameters
    add_arg("--seed", type=int, default=42, help="Random seed for reproducable results")
    add_arg("--device", type=str, default="0", help="Device used for training. Can be `1, 2` or `-1`.")
    add_arg("--batch_size", type=int, default=32, help="Number of spectagams in stack")
    # add_arg(
    #     "--model_params", 
    #     type=eval, 
    #     default={"vec_len" : 128}, 
    #     help="Additional model params as kwargs"
    # )

    add_arg("--datasets",default=["cifar10"],type=str,nargs="+", \
        help="Datasets to use for training. Default is only CIFAR10")
    
    add_arg("--lr", type=float, default=0.0001, help="Learning rate")
    add_arg("--adam_b1", type=float, default=0.5, help="Adam B1 parameter")
    add_arg("--adam_b2", type=float, default=0.9, help="Adam B2 parameter")

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    # add timestamp to name and create this run folder
    # timestamp = get_timestamp()
    # name = args.name + "_" + timestamp if args.name else timestamp
    return args

