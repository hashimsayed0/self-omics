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


param = util.parse_arguments(train_in_phases=True)
util.set_seeds(param.seed)   
param.train_in_phases = True 
config = vars(param)
fold = param.fold_idx
abc_dm = datamodules.ABCDataModule(fold, **config)
checkpoint_path = os.path.join(param.checkpoints_dir, param.exp_name, 'fold-{}'.format(fold))

if param.resume_training:
    model = lit_models.Comics.load_from_checkpoint(param.model_path)
else:
    A_shape, B_shape, C_shape = util.compute_input_shapes(abc_dm)
    config['input_size_A'] = A_shape
    config['input_size_B'] = B_shape
    config['input_size_C'] = C_shape
    config['class_weights'] = abc_dm.class_weights
    model = lit_models.Comics(current_phase='p1', **config)

early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_p1(param, checkpoint_path, fold)
trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
trainer.fit(model, abc_dm)
p1_model_path = model_checkpoint.best_model_path
wandb.finish()

abc_dm.phase = 'p2'
# abc_dm.setup(stage='fit')
early_stopping, model_checkpoint, csv_logger, wandb_logger = util.define_callbacks_loggers_p2(param, checkpoint_path, fold)
model = None
if param.cs_p1_max_epochs > 0:
    model = lit_models.Comics.load_from_checkpoint(p1_model_path, current_phase='p2')
else:
    model = lit_models.Comics(current_phase='p2', **config)
trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
trainer.fit(model, abc_dm)
p2_model_path = model_checkpoint.best_model_path
wandb.finish()
    
abc_dm.mode = 'downstream'
abc_dm.phase = 'p3'
# abc_dm.setup(stage='fit')
early_stopping, model_checkpoint, csv_logger, wandb_logger = util.define_callbacks_loggers_p3(param, checkpoint_path, fold)
model = None
if param.cs_p2_max_epochs > 0:
    model = lit_models.Comics.load_from_checkpoint(p2_model_path, current_phase='p3')
elif param.cs_p1_max_epochs > 0:
    model = lit_models.Comics.load_from_checkpoint(p1_model_path, current_phase='p3')
else:
    model = lit_models.Comics(current_phase='p3', **config)
trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
trainer.fit(model, abc_dm)
trainer.test(datamodule=abc_dm, ckpt_path='best')



