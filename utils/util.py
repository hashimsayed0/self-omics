import argparse
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
import torch
import numpy as np
from models.lit_models import AutoEncoder, DownstreamModel, ComicsModel, Comics
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
from .datamodules import ABCDataModule

def parse_arguments(run_in_phases=False):
    parser = argparse.ArgumentParser(description='Runs the specified command')

    # general arguments
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models, settings and intermediate results are saved in folder in this directory')
    parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
    parser.add_argument("--fold_idx", type=int, default=0, 
                            help="fold_idx for k-fold cross validation")
    parser.add_argument("--num_folds", type=int, default=5, 
                            help="number of folds for k-fold cross validation if one_fold is set to False")
    parser.add_argument("--class_0_weight", type=float, default=0.5, 
                            help="weight of class 0 in the loss function")
    parser.add_argument("--plot_confusion_matrix", default=False, type=lambda x: (str(x).lower() == 'true'),
                            help="plot confusion matrix")
    parser.add_argument("--prediction_data", type=str, default="test",
                            help="data to predict on, options: test, train, val, all")
    
    # trainer related arguments
    parser.add_argument("--exp_name", type=str, default="test")
    
    parser = ABCDataModule.add_data_module_args(parser)
    
    if run_in_phases:
        parser = Comics.add_model_specific_args(parser)
    else:
        parser = AutoEncoder.add_model_specific_args(parser)
        parser = DownstreamModel.add_model_specific_args(parser)

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
            callback_key = 'val_recon_all_loss'
    elif param.mask_A:
        if param.ae_net == 'vae':
            callback_key = 'val_recon_A_kl_loss'
        elif param.ae_net == 'ae':
            if param.recon_all_thrice:
                callback_key = 'val_total_recon_all_loss'   
            else: 
                callback_key = 'val_recon_all_loss'
    else:
        callback_key = 'val_recon_loss'
    if param.cont_align_loss_criterion != 'none' or param.cont_noise_loss_criterion != 'none' or param.add_distance_loss:
        callback_key = 'val_pretext_loss'
    
    param.max_epochs = param.pretraining_max_epochs
    param.min_epochs = 0
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='pretraining')
    early_stopping = EarlyStopping(callback_key, patience=param.pretraining_patience)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='min', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}'.format(param.exp_name), name = 'fold-{f}-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, wandb_logger, csv_logger


def define_callbacks_loggers_downstream(param, checkpoint_path, count):
    early_stopping_key = 'val_{}_loss'.format(param.ds_task)
    if param.ds_task == 'class':
        callback_key = 'val_{}'.format(param.ds_class_callback_key)
    elif param.ds_task == 'surv':
        callback_key = 'val_{}'.format(param.ds_surv_callback_key)
    elif param.ds_task == 'reg':
        callback_key = 'val_{}'.format(param.ds_reg_callback_key)
    elif param.ds_task == 'multi':
        callback_key = 'val_down_loss'
        early_stopping_key = 'val_down_loss'
    param.max_epochs = param.downstream_max_epochs
    param.min_epochs = 0
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='downstream')
    early_stopping = EarlyStopping(early_stopping_key, patience=param.downstream_patience)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='max', save_top_k=1)
    # wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-downstream'.format(param.exp_name), name = 'fold-{f}-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    # return early_stopping, model_checkpoint, wandb_logger, csv_logger
    return early_stopping, model_checkpoint, csv_logger

def define_callbacks_loggers_p1(param, checkpoint_path, count):
    if param.mask_B:
        if param.ae_net == 'vae':
            callback_key = 'val_recon_B_kl_loss'
        elif param.ae_net == 'ae':
            callback_key = 'val_recon_all_loss'
    elif param.mask_A:
        if param.ae_net == 'vae':
            callback_key = 'val_recon_A_kl_loss'
        elif param.ae_net == 'ae':
            if param.recon_all_thrice:
                callback_key = 'val_total_recon_all_loss'   
            else: 
                callback_key = 'val_recon_all_loss'
    else:
        callback_key = 'val_recon_loss'
    if param.cont_align_loss_criterion != 'none' or param.cont_noise_loss_criterion != 'none' or param.add_distance_loss:
        callback_key = 'val_pretext_loss'
    
    param.max_epochs = param.cs_p1_max_epochs
    param.min_epochs = 0
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='p1')
    early_stopping = EarlyStopping(callback_key, patience=param.cs_p1_patience, verbose=True)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='min', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-p1'.format(param.exp_name), name = 'fold-{f}-p1-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, wandb_logger, csv_logger


def define_callbacks_loggers_p2(param, checkpoint_path, count):
    early_stopping_key = 'val_{}_loss'.format(param.ds_task)
    callback_key = 'val_{}_loss'.format(param.ds_task)
    # if param.ds_task == 'class':
    #     callback_key = 'val_{}'.format(param.ds_class_callback_key)
    # elif param.ds_task == 'surv':
    #     callback_key = 'val_{}'.format(param.ds_surv_callback_key)
    # elif param.ds_task == 'reg':
    #     callback_key = 'val_{}'.format(param.ds_reg_callback_key)
    # elif param.ds_task == 'multi':
    #     callback_key = 'val_down_loss'
    #     early_stopping_key = 'val_down_loss'
    if param.ds_task == 'multi':
        callback_key = 'val_down_loss'
        early_stopping_key = 'val_down_loss'
    param.max_epochs = param.cs_p2_max_epochs
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='p2')
    early_stopping = EarlyStopping(early_stopping_key, patience=param.cs_p2_patience, verbose=True)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='min', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-p2'.format(param.exp_name), name = 'fold-{f}-p2-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, csv_logger, wandb_logger

def define_callbacks_loggers_p3(param, checkpoint_path, count):
    early_stopping_key = 'val_total_loss'
    if param.ds_task == 'class':
        callback_key = 'val_{}'.format(param.ds_class_callback_key)
    elif param.ds_task == 'surv':
        callback_key = 'val_{}'.format(param.ds_surv_callback_key)
    elif param.ds_task == 'reg':
        callback_key = 'val_{}'.format(param.ds_reg_callback_key)
    elif param.ds_task == 'multi':
        callback_key = 'val_down_loss'
    param.max_epochs = param.cs_p3_max_epochs
    csv_logger = pl_loggers.CSVLogger(checkpoint_path, name='p3')
    early_stopping = EarlyStopping(early_stopping_key, patience=param.cs_p3_patience, verbose=True)
    model_checkpoint = ModelCheckpoint(csv_logger.log_dir, monitor=callback_key, mode='max', save_top_k=1)
    wandb_logger = pl_loggers.WandbLogger(project = 'tcga_contrastive', group = '{}-p3'.format(param.exp_name), name = 'fold-{f}-p3-v{v}'.format(f=count, v=csv_logger.version), offline=False)
    return early_stopping, model_checkpoint, csv_logger, wandb_logger

def save_latents(outputs, pred_data, latent_dir):
    sample_ids_list = []
    for x in outputs:
        sample_ids_list.extend(x["sample_ids"])
    h_concat = torch.cat([x["h"] for x in outputs]).cpu().numpy()
    latent_space = pd.DataFrame(h_concat, index=sample_ids_list)
    # check if dir exists, else create
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)
    latent_space.to_csv(os.path.join(latent_dir, '{}_latent_space.tsv'.format(pred_data)), sep='\t')

def save_model_outputs(outputs, pred_data, ds_model_output_dir):
    sample_ids_list = []
    for x in outputs:
        sample_ids_list.extend(x["sample_ids"])
    y_out_concat = torch.cat([x["y_out"] for x in outputs]).cpu().numpy()
    model_outputs = pd.DataFrame(y_out_concat, index=sample_ids_list)
    # check if dir exists, else create
    if not os.path.exists(ds_model_output_dir):
        os.makedirs(ds_model_output_dir)
    model_outputs.to_csv(os.path.join(ds_model_output_dir, '{}_model_outputs.tsv'.format(pred_data)), sep='\t')