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
abc_dm.mode = 'downstream'
# checkpoint_path = os.path.join(param.checkpoints_dir, param.exp_name, 'fold-{}'.format(fold))
# early_stopping, model_checkpoint, wandb_logger, csv_logger = util.define_callbacks_loggers_pretraining(param, checkpoint_path, fold)
# early_stopping, model_checkpoint, csv_logger = util.define_callbacks_loggers_downstream(param, checkpoint_path, fold)
# if param.load_pretrained_ds:
#     classifier = lit_models.DownstreamModel.load_from_checkpoint(param.pretrained_ds_path, param.pretrained_ae_path, abc_dm.class_weights)
# else:
#     classifier = lit_models.DownstreamModel(param.pretrained_ae_path, abc_dm.class_weights, **vars(param))
A_shape, B_shape, C_shape = util.compute_input_shapes(abc_dm)
classifier = lit_models.DownstreamModel(param.pretrained_ae_path, A_shape, B_shape, C_shape, abc_dm.class_weights, **vars(param))
latent_save_path = os.path.join(os.path.dirname(param.pretrained_ae_path), 'latents')
# if param.ds_save_latent_dataset:
#     classifier.feature_extractor.freeze()
#     for p in classifier.class_net.parameters():
#         p.requires_grad = False
#     param.max_epochs = 1
classifier_trainer = Trainer.from_argparse_args(param, resume_from_checkpoint=param.pretrained_ds_path)

if param.ds_save_latent_pred:
    if param.prediction_data == 'all':
        for pred_data in ['train', 'val', 'test']:
            abc_dm.prediction_data = pred_data
            outputs = classifier_trainer.predict(classifier, datamodule=abc_dm)
            util.save_latents(outputs, pred_data, latent_save_path)
    else:
        outputs = classifier_trainer.predict(classifier, datamodule=abc_dm)
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