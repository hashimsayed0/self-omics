from cProfile import label
import numpy as np
import pandas as pd
import os

from wandb import wandb
from utils import util, datamodules
from models import lit_models
from pytorch_lightning import Trainer


param = util.parse_arguments()
util.set_seeds(param.seed)

for fold in range(param.num_folds):
    print('Fold ' + str(fold))
    if param.one_fold:
        if param.fold_idx != fold:
            continue
    
    abc_dm = datamodules.ABCDataModule(param, fold)
    dict_args = vars(param)
    A_shape, B_shape, C_shape = util.compute_input_shapes(abc_dm)
    ae = lit_models.AutoEncoder(A_shape, B_shape, C_shape, **dict_args)
    early_stopping, model_checkpoint, wandb_logger, csv_logger, checkpoint_path = util.define_callbacks_loggers_pretraining(param, fold)
    ae_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
    ae_trainer.fit(ae, abc_dm)
    wandb.finish()

    classifier = lit_models.Classifier(model_checkpoint.best_model_path, abc_dm.class_weights, **dict_args)
    early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
    classifier_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
    classifier_trainer.fit(classifier, abc_dm)
    wandb.finish()