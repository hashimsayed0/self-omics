import argparse
from pytorch_lightning import Trainer, seed_everything
import torch
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Runs the specified command')

    # general arguments
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models, settings and intermediate results are saved in folder in this directory')
    parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
    parser.add_argument("--one_fold", action="store_true",
                            help="to use only one fold for training, by default k-fold cross validation is done")
    parser.add_argument("--fold_idx", type=int, default=0, 
                            help="if one_fold is set to True, this is the fold_idx number")
    parser.add_argument("--num_folds", type=int, default=5, 
                            help="number of folds for k-fold cross validation if one_fold is set to False")
    parser.add_argument("--class_0_weight", type=float, default=0.5, 
                            help="weight of class 0 in the loss function")

    # data arguments
    parser.add_argument("--data_dir", type=str, default="data/ABC_inter",
                            help="directory containing the dataset")
    parser.add_argument('--use_sample_list', action='store_true',
                            help='provide a subset sample list of the dataset, store in the path data_dir/sample_list.tsv, if False use all the samples')
    parser.add_argument('--batch_size', type=int, default=32,
                            help='data batch size')
    parser.add_argument('--train_val_split', type=float, default=0.9,
                            help='train/val split ratio')
    parser.add_argument('--num_workers', type=int, default=0,
                            help='number of workers for data loading')
    
    
    # trainer related arguments
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--pretraining_patience", type=int, default=10)
    parser.add_argument("--downstream_patience", type=int, default=10)
    
    parser = VAE.add_model_specific_args(parser)
    parser = Classifier.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    param = parser.parse_args()
    return param
    
    return parser.parse_args()


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)