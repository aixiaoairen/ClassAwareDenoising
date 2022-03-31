# @Time : 2022/3/31 17:51
# @Author : LiangHao
# @File : option.py
import os
import argparse

def option():
    parser = argparse.ArgumentParser()
    # Path to Save some files
    parser.add_argument('--model_dir', type=str, default=".\Model",
                        help='The Path to save training model')
    parser.add_argument('--data_dir', type=str, default="TestMethod\data")
    parser.add_argument('--datasets', type=dict, default={
        "JPEGImages": "*.jpg"
    },
                        help="The Name and type of DataSets")
    # Make DataSet and Load data to memory
    parser.add_argument('--patch_size', type=int, default=128,
                        help="The Size of Cropping Images (default: 128)")
    parser.add_argument('--num_worker', type=int, default=1,
                        help="The Number of Process")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch Size of Training (default: 64)")
    parser.add_argument('--peak', type=float, default=4.0, help="the peak value of poisson noise")
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Initialized Learning Rate (default: 1e-4)")
    # Training
    parser.add_argument('--in_channel', type=int, default=1,
                        help="The Numbers of Channels")
    parser.add_argument('--wf', type=int, default=63,
                        help="The Num of Convolution Kernel")
    parser.add_argument('--epochs', type=int, default=120,
                        help="Train Epochs (default: 120)")
    parser.add_argument('--resume', type=bool, default=False,
                        help="Judge Whether The Training")

    args = parser.parse_args()
    return args


