import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from wandb import wandb
from utils import util, datamodules
from models import lit_models
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torch

param = util.parse_arguments()
util.set_seeds(param.seed)

fold = 0
dict_args = vars(param)
abc_dm = datamodules.ABCDataModule(fold, **dict_args)
checkpoint_path = 'checkpoints/010/fold-0'
ae_model_path = './checkpoints/test/fold-0/pretraining/version_58/epoch=19-step=239.ckpt'
ds_model_path = './checkpoints/test/fold-0/downstream/version_25/epoch=14-step=179.ckpt'

early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
# classifier = lit_models.DownstreamModel(ae_model_path, abc_dm.class_weights, **vars(param))
classifier_trainer = Trainer.from_argparse_args(param, callbacks=[early_stopping, model_checkpoint], logger=[csv_logger, wandb_logger])
loaded_model = lit_models.DownstreamModel.load_from_checkpoint(ds_model_path)
# classifier_trainer.test(classifier, datamodule=abc_dm, ckpt_path=ds_model_path)
outputs = classifier_trainer.predict(loaded_model, datamodule=abc_dm)
if param.ds_task == 'class':
    y_true_binary = torch.cat([x["y_true"] for x in outputs]).cpu().numpy()
    y_true = np.argmax(y_true_binary, axis=1)
    y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().numpy()
    y_prob = torch.cat([x["y_prob"] for x in outputs]).cpu().numpy()
    preds = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_true_binary": y_true_binary
    }

elif param.ds_task == "surv":
    y_true_E = torch.cat([x["y_true_E"] for x in outputs]).cpu().numpy()
    y_true_T = torch.cat([x["y_true_T"] for x in outputs]).cpu().numpy()
    y_pred_risk = torch.cat([x["risk"] for x in outputs]).cpu().numpy()
    y_pred_survival = torch.cat([x["survival"] for x in outputs]).cpu().numpy()
    preds = {
        "y_true_E": y_true_E,
        "y_true_T": y_true_T,
        "y_pred_risk": y_pred_risk,
        "y_pred_survival": y_pred_survival
    }
cf_matrix = confusion_matrix(preds['y_true'], preds['y_pred'])
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in abc_dm.classes],
                    columns = [i for i in abc_dm.classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')