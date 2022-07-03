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

param = util.parse_arguments(run_in_phases=True)
util.set_seeds(param.seed)
param.train_in_phases = True 
fold = 0
dict_args = vars(param)
abc_dm = datamodules.ABCDataModule(fold, **dict_args)
abc_dm.mode = 'downstream'
abc_dm.phase = 'p2'
model = lit_models.Comics.load_from_checkpoint(param.model_path)
latent_save_path = os.path.join(os.path.dirname(param.model_path), 'latents')
trainer = Trainer.from_argparse_args(param)

if param.ds_save_latent_dataset:
    if param.prediction_data == 'all':
        for pred_data in ['train', 'val', 'test']:
            abc_dm.prediction_data = pred_data
            outputs = trainer.predict(model, datamodule=abc_dm)
            util.save_latents(outputs, pred_data, latent_save_path)
    else:
        outputs = trainer.predict(model, datamodule=abc_dm)
        util.save_latents(outputs, param.prediction_data, latent_save_path)

if param.plot_confusion_matrix:
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