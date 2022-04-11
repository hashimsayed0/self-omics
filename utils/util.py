import argparse
from pytorch_lightning import Trainer, seed_everything
import torch
import numpy as np
from models.lit_models import AutoEncoder, Classifier
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
from .datamodules import ABCDataModule

def parse_arguments():
    parser = argparse.ArgumentParser(description='Runs the specified command')

    # general arguments
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models, settings and intermediate results are saved in folder in this directory')
    parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
    parser.add_argument("--one_fold", default=False, type=lambda x: (str(x).lower() == 'true'), 
                            help="to use only one fold for training, by default k-fold cross validation is done")
    parser.add_argument("--fold_idx", type=int, default=0, 
                            help="if one_fold is set to True, this is the fold_idx number")
    parser.add_argument("--num_folds", type=int, default=5, 
                            help="number of folds for k-fold cross validation if one_fold is set to False")
    parser.add_argument("--class_0_weight", type=float, default=0.5, 
                            help="weight of class 0 in the loss function")   
    parser.add_argument("--downstream_task", type=str, default='ctc', 
                            help='options: ctc (cancer type classification), sa (survival analysis)')
    
    # trainer related arguments
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--pretraining_patience", type=int, default=10)
    parser.add_argument("--downstream_patience", type=int, default=10)
    parser.add_argument("--pretraining_max_epochs", type=int, default=15)
    parser.add_argument("--downstream_max_epochs", type=int, default=75)
    
    parser = ABCDataModule.add_data_module_args(parser)
    parser = AutoEncoder.add_model_specific_args(parser)
    parser = Classifier.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    param = parser.parse_args()
    return param


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)

def compute_input_shapes(abc_dm):
    if isinstance(abc_dm.A_df, list):
        A_shape = [A_ch.shape[0] for A_ch in abc_dm.A_df]
    else:
        A_shape = abc_dm.A_df.shape[0]
    if isinstance(abc_dm.B_df, list):
        B_shape = [B_ch.shape[0] for B_ch in abc_dm.B_df]
    else:
        B_shape = abc_dm.B_df.shape[0]
    C_shape = abc_dm.C_df.shape[0]
    return A_shape, B_shape, C_shape

def define_callbacks_loggers_pretraining(param, checkpoint_path, count):
    if param.mask_B:
        if param.ae_net == 'vae':
            callback_key = 'val_recon_B_kl_loss'
        elif param.ae_net == 'ae':
            callback_key = 'val_recon_B_loss'
    elif param.mask_A:
        if param.ae_net == 'vae':
            callback_key = 'val_recon_A_kl_loss'
        elif param.ae_net == 'ae':
            callback_key = 'val_recon_A_loss'
    elif param.cont_loss != 'none':
        callback_key = 'val_pretext_loss'
    else:
        callback_key = 'val_recon_loss'
    param.max_epochs = param.pretraining_max_epochs
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='pretraining')
    early_stopping = EarlyStopping(callback_key, patience=param.pretraining_patience)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='min', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-pretraining'.format(param.exp_name), name = 'fold-{f}-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, wandb_logger, csv_logger


def define_callbacks_loggers_downstream(param, checkpoint_path, count):
    param.max_epochs = param.downstream_max_epochs
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='downstream')
    early_stopping = EarlyStopping('val_down_loss', patience=param.downstream_patience)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor='val_accuracy', mode='max', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-downstream'.format(param.exp_name), name = 'fold-{f}-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, wandb_logger, csv_logger

