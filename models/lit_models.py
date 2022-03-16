from pyrsistent import v
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import Encoder, Decoder, ProjectionHead, AENet, ClassifierNet
from .losses import SimCLR_Loss, weighted_binary_cross_entropy
from torchmetrics.functional import f1_score, auroc, precision, recall, accuracy
import wandb


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, latent_size, ae_lr, ae_weight_decay, ae_momentum, ae_drop_p, loss_temperature, split_B, batch_size, ae_dim_1B, ae_dim_2B, ae_dim_1A, ae_dim_2A, ae_dim_1C, ae_dim_2C, **config):
        super(AutoEncoder, self).__init__()
        self.input_size_A = input_size_A
        self.input_size_B = input_size_B
        self.input_size_C = input_size_C
        self.latent_size = latent_size
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay
        self.ae_momentum = ae_momentum
        self.ae_drop_p = ae_drop_p
        # self.aenet = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=ae_drop_p, dim_1B=16, dim_2B=256, dim_1A=512, dim_2A=256, dim_1C=512, dim_2C=256)
        self.aenet = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        self.projector = ProjectionHead(latent_size, latent_size, latent_size // 2)
        self.cont_criterion = SimCLR_Loss(batch_size = batch_size, temperature = loss_temperature)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument("--latent_size", type=int, default=128)
        parser.add_argument("--ae_lr", type=float, default=1e-3)
        parser.add_argument("--ae_weight_decay", type=float, default=0.0001)
        parser.add_argument("--ae_momentum", type=float, default=0.9)
        parser.add_argument("--ae_drop_p", type=float, default=0.2)
        parser.add_argument("--loss_temperature", type=float, default=0.5)
        parser.add_argument("--ae_dim_1B", type=int, default=16)
        parser.add_argument("--ae_dim_2B", type=int, default=32)
        parser.add_argument("--ae_dim_1A", type=int, default=32)
        parser.add_argument("--ae_dim_2A", type=int, default=32)
        parser.add_argument("--ae_dim_1C", type=int, default=32)
        parser.add_argument("--ae_dim_2C", type=int, default=32)
        return parent_parser

    def forward(self, x):
        return self.aenet.encode(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay)
    
    def training_step(self, batch, batch_idx):
        x_A, x_B, x_C, _ = batch
        h_A, h_B, h_C = self.aenet.encode((x_A, x_B, x_C))
        x_A_recon, x_B_recon, x_C_recon = self.aenet.decode((h_A, h_B, h_C))
        x_B_recon_loss = []
        for i in range(len(x_B)):
            x_B_recon_loss.append(F.mse_loss(x_B_recon[i], x_B[i]))
        recon_loss = F.mse_loss(x_A_recon, x_A) + sum(x_B_recon_loss) + F.mse_loss(x_C_recon, x_C)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True)
        z_A = self.projector(h_A)
        z_B = self.projector(h_B)
        z_C = self.projector(h_C)
        cont_loss = self.cont_criterion(z_A, z_B) + self.cont_criterion(z_B, z_C) + self.cont_criterion(z_C, z_A)
        pretext_loss = recon_loss + cont_loss
        self.log("train_cont_loss", cont_loss, on_step=True, on_epoch=True)
        self.log("train_pretext_loss", pretext_loss, on_step=True, on_epoch=True)
        return pretext_loss
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_pretext_loss', summary='min')
        x_A, x_B, x_C, _ = batch
        h_A, h_B, h_C = self.aenet.encode((x_A, x_B, x_C))
        x_A_recon, x_B_recon, x_C_recon = self.aenet.decode((h_A, h_B, h_C))
        x_B_recon_loss = []
        for i in range(len(x_B)):
            x_B_recon_loss.append(F.mse_loss(x_B_recon[i], x_B[i]))
        recon_loss = F.mse_loss(x_A_recon, x_A) + sum(x_B_recon_loss) + F.mse_loss(x_C_recon, x_C)
        z_A = self.projector(h_A)
        z_B = self.projector(h_B)
        z_C = self.projector(h_C)
        cont_loss = self.cont_criterion(z_A, z_B) + self.cont_criterion(z_B, z_C) + self.cont_criterion(z_C, z_A)
        pretext_loss = recon_loss + cont_loss
        return {
            'val_recon_loss': recon_loss,
            "val_cont_loss": cont_loss,
            "val_pretext_loss": pretext_loss
        }

    def validation_epoch_end(self, outputs):
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_cont_loss = torch.stack([x["val_cont_loss"] for x in outputs]).mean()
        avg_pretext_loss = torch.stack([x["val_pretext_loss"] for x in outputs]).mean()
        self.log("val_recon_loss", avg_recon_loss)
        self.log("val_cont_loss", avg_cont_loss)
        self.log("val_pretext_loss", avg_pretext_loss)
    

class Classifier(pl.LightningModule):
    def __init__(self, path_to_model, class_weights, num_classes, latent_size, cl_lr, cl_weight_decay, cl_momentum, cl_drop_p, **config):
        super(Classifier, self).__init__()
        self.input_size = latent_size
        self.cl_drop_p = cl_drop_p
        self.num_classes = num_classes
        self.cl_lr = cl_lr
        self.cl_weight_decay = cl_weight_decay
        self.cl_momentum = cl_momentum
        self.class_weights = class_weights
        self.wbce = weighted_binary_cross_entropy
        self.criterion = nn.CrossEntropyLoss()
        self.feature_extractor = AutoEncoder.load_from_checkpoint(path_to_model)
        self.classifier = ClassifierNet(num_classes, latent_size, dropout_p=cl_drop_p)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--cl_drop_p", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--cl_lr", type=float, default=1e-3)
        parser.add_argument("--cl_weight_decay", type=float, default=0.0001)
        parser.add_argument("--cl_momentum", type=float, default=0.9)
        return parent_parser

    def forward(self, x):
        h_A, h_B, h_C = self.feature_extractor(x)
        h = torch.mean(torch.stack([h_A, h_B, h_C]), axis=0)
        return self.classifier(h)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cl_lr, weight_decay=self.cl_weight_decay)
    
    def training_step(self, batch, batch_idx):
        x_A, x_B, x_C, y = batch
        y_pred = self.forward((x_A, x_B, x_C))
        # down_loss = self.wbce(y_pred, y, self.class_weights)
        down_loss = self.criterion(y_pred, y)
        y = y.long()
        acc = accuracy(y_pred, y)
        self.log("train_down_loss", down_loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True)
        return down_loss
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_accuracy', summary='max')
        x_A, x_B, x_C, y = batch
        y_pred = self.forward((x_A, x_B, x_C))
        # down_loss = self.wbce(y_pred, y, self.class_weights)
        down_loss = self.criterion(y_pred, y)
        y = y.long()
        acc = accuracy(y_pred, y)
        prec = precision(y_pred, y)
        rec = recall(y_pred, y)
        f1 = f1_score(y_pred, y)
        auc = auroc(y_pred, y, pos_label=1)
        return {
            "val_down_loss": down_loss,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_auc": auc
        }

    def validation_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["val_down_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        avg_prec = torch.stack([x["val_precision"] for x in outputs]).mean()
        avg_rec = torch.stack([x["val_recall"] for x in outputs]).mean()
        avg_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()
        avg_auc = torch.stack([x["val_auc"] for x in outputs]).mean()
        self.log("val_down_loss", avg_down_loss)
        self.log("val_accuracy", avg_acc)
        self.log("val_precision", avg_prec)
        self.log("val_recall", avg_rec)
        self.log("val_f1", avg_f1)
        self.log("val_auc", avg_auc)