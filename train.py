import numpy as np
import pandas as pd
import seaborn as sn
import os
from sklearn.metrics import confusion_matrix
from wandb import wandb
from utils import util, datamodules
from models import lit_models
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt


param = util.parse_arguments()
util.set_seeds(param.seed)

for fold in range(param.num_folds):
    print('Fold ' + str(fold))
    if param.one_fold:
        if param.fold_idx != fold:
            continue
    
    dict_args = vars(param)
    abc_dm = datamodules.ABCDataModule(fold, **dict_args)
    checkpoint_path = os.path.join(param.checkpoints_dir, param.exp_name, 'fold-{}'.format(fold))
    early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_pretraining(param, checkpoint_path, fold)

    if param.load_pretrained_ae:
        ae_model_path = param.pretrained_ae_path
    else:
        A_shape, B_shape, C_shape = util.compute_input_shapes(abc_dm)
        ae = lit_models.AutoEncoder(A_shape, B_shape, C_shape, **vars(param))
        ae_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
        ae_trainer.fit(ae, abc_dm)
        ae_model_path = model_checkpoint.best_model_path
        wandb.finish()
        
    early_stopping, model_checkpoint, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
    classifier = lit_models.DownstreamModel(ae_model_path, abc_dm.class_weights, checkpoint_path, **vars(param))
    classifier_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
    classifier_trainer.fit(classifier, abc_dm)
    classifier_trainer.test(datamodule=abc_dm, ckpt_path='best')
