from cProfile import label
import numpy as np
import pandas as pd
import os
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
    ae = lit_models.AutoEncoder(abc_dm.A_df.shape[0], abc_dm.B_df.shape[0], abc_dm.C_df.shape[0], dict_args)
    early_stopping, model_checkpoint, wandb_logger, csv_logger, checkpoint_path = util.define_callbacks_loggers_pretraining(param, fold)
    trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
    trainer.fit(ae, abc_dm)