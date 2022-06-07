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
config = vars(param)
fold = param.fold_idx
abc_dm = datamodules.ABCDataModule(fold, **config)
checkpoint_path = os.path.join(param.checkpoints_dir, param.exp_name, 'fold-{}'.format(fold))
early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_pretraining(param, checkpoint_path, fold)

if param.load_pretrained_ae:
    ae_model_path = param.pretrained_ae_path
else:
    A_shape, B_shape, C_shape = util.compute_input_shapes(abc_dm)
    ae_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
    config['input_size_A'] = A_shape
    config['input_size_B'] = B_shape
    config['input_size_C'] = C_shape
    ae = lit_models.AutoEncoder(**config)
    ae_trainer.fit(ae, abc_dm)
    ae_model_path = model_checkpoint.best_model_path
    
early_stopping, model_checkpoint, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
config['ae_model_path'] = ae_model_path
config['class_weights'] = abc_dm.class_weights
classifier = lit_models.DownstreamModel(**config)
classifier_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
classifier_trainer.fit(classifier, abc_dm)
# classifier_trainer.test(datamodule=abc_dm, ckpt_path='best')

abc_dm.mode = 'downstream'
abc_dm.setup('fit')
ds_model_path = model_checkpoint.best_model_path
early_stopping, model_checkpoint, csv_logger = util.define_callbacks_loggers_comics(param, checkpoint_path, fold)
config['ds_model_path'] = ds_model_path
comics_model = lit_models.ComicsModel(**config)
comics_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
comics_trainer.fit(comics_model, abc_dm)
comics_trainer.test(datamodule=abc_dm, ckpt_path='best')



