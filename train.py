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
dict_args = vars(param)
fold = param.fold_idx
abc_dm = datamodules.OmicsDataModule(fold, **dict_args)
checkpoint_path = os.path.join(param.checkpoints_dir, param.exp_name, 'fold-{}'.format(fold))
early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_pretraining(param, checkpoint_path, fold)

if param.load_pretrained_ae:
    ae_model_path = param.pretrained_ae_path
else:
    ABC_shapes = util.compute_input_shapes(abc_dm)
    if param.pretraining_max_epochs > 0:
        ae_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
        ae = lit_models.AutoEncoder(ABC_shapes, **vars(param))
        ae_trainer.fit(ae, abc_dm)
        ae_model_path = model_checkpoint.best_model_path
        # wandb.finish()
    else:
        ae_model_path = None
    
early_stopping, model_checkpoint, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
classifier = lit_models.DownstreamModel(ae_model_path, ABC_shapes, abc_dm.class_weights, **vars(param))
classifier_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
abc_dm.mode = 'downstream'
classifier_trainer.fit(classifier, abc_dm)
wandb.config.update({'ds_model_path': model_checkpoint.best_model_path}, allow_val_change=True)
if param.downstream_max_epochs > 0:
    classifier_trainer.test(datamodule=abc_dm, ckpt_path='best')
else:
    classifier_trainer.test(model=classifier, datamodule=abc_dm)