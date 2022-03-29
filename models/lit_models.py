import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import SimCLRProjectionHead, CLIPProjectionHead, AENet, ClassifierNet
from .losses import SimCLR_Loss, weighted_binary_cross_entropy, CLIPLoss
from torchmetrics.functional import f1_score, auroc, precision, recall, accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, latent_size, projection_size, ae_lr, ae_weight_decay, ae_momentum, ae_drop_p, cont_loss, cont_loss_temp, cont_loss_weight, split_B, batch_size, ae_dim_1B, ae_dim_2B, ae_dim_1A, ae_dim_2A, ae_dim_1C, ae_dim_2C, **config):
        super(AutoEncoder, self).__init__()
        self.input_size_A = input_size_A
        self.input_size_B = input_size_B
        self.input_size_C = input_size_C
        self.latent_size = latent_size
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay
        self.ae_momentum = ae_momentum
        self.ae_drop_p = ae_drop_p
        self.cont_loss_weight = cont_loss_weight
        # self.aenet = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=ae_drop_p, dim_1B=16, dim_2B=256, dim_1A=512, dim_2A=256, dim_1C=512, dim_2C=256)
        self.aenet = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        if cont_loss == "simclr":
            self.projector = SimCLRProjectionHead(latent_size, latent_size, latent_size // 2)
            self.cont_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp)
        elif cont_loss == "clip":
            self.projector = CLIPProjectionHead(latent_size, projection_size, ae_drop_p)
            self.cont_criterion = CLIPLoss(temperature = cont_loss_temp, latent_size = latent_size, proj_size = projection_size)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument("--latent_size", type=int, default=512)
        parser.add_argument("--projection_size", type=int, default=256)
        parser.add_argument("--ae_lr", type=float, default=1e-3)
        parser.add_argument("--ae_weight_decay", type=float, default=0.0001)
        parser.add_argument("--ae_momentum", type=float, default=0.9)
        parser.add_argument("--ae_drop_p", type=float, default=0.2)
        parser.add_argument("--cont_loss", type=str, default="simclr", help="contrastive loss to use, options: simclr, clip")
        parser.add_argument("--cont_loss_temp", type=float, default=0.5)
        parser.add_argument("--cont_loss_weight", type=float, default=0.01)
        parser.add_argument("--ae_dim_1B", type=int, default=16)
        parser.add_argument("--ae_dim_2B", type=int, default=32)
        parser.add_argument("--ae_dim_1A", type=int, default=32)
        parser.add_argument("--ae_dim_2A", type=int, default=32)
        parser.add_argument("--ae_dim_1C", type=int, default=32)
        parser.add_argument("--ae_dim_2C", type=int, default=32)
        parser.add_argument("--load_pretrained_ae", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--pretrained_ae_path", type=str, default="")
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
        # z_A = self.projector(h_A)
        # z_B = self.projector(h_B)
        # z_C = self.projector(h_C)
        # cont_loss = self.cont_criterion(z_A, z_B) + self.cont_criterion(z_B, z_C) + self.cont_criterion(z_C, z_A)

        loss_A_B, loss_A1, loss_B1 = self.cont_criterion(h_A, h_B)
        loss_B_C, loss_B2, loss_C1 = self.cont_criterion(h_B, h_C)
        loss_C_A, loss_C2, loss_A2 = self.cont_criterion(h_C, h_A)
        cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
        loss_A = (loss_A1 + loss_A2) / 2
        loss_B = (loss_B1 + loss_B2) / 2
        loss_C = (loss_C1 + loss_C2) / 2
        pretext_loss = recon_loss + self.cont_loss_weight * cont_loss
        self.log("train_cont_loss_A", loss_A, on_step=True, on_epoch=True)
        self.log("train_cont_loss_B", loss_B, on_step=True, on_epoch=True)
        self.log("train_cont_loss_C", loss_C, on_step=True, on_epoch=True)
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
        # z_A = self.projector(h_A)
        # z_B = self.projector(h_B)
        # z_C = self.projector(h_C)
        # cont_loss = self.cont_criterion(z_A, z_B) + self.cont_criterion(z_B, z_C) + self.cont_criterion(z_C, z_A)

        loss_A_B, loss_A1, loss_B1 = self.cont_criterion(h_A, h_B)
        loss_B_C, loss_B2, loss_C1 = self.cont_criterion(h_B, h_C)
        loss_C_A, loss_C2, loss_A2 = self.cont_criterion(h_C, h_A)
        cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
        loss_A = (loss_A1 + loss_A2) / 2
        loss_B = (loss_B1 + loss_B2) / 2
        loss_C = (loss_C1 + loss_C2) / 2
        pretext_loss = recon_loss + self.cont_loss_weight * cont_loss
        return {
            'val_recon_loss': recon_loss,
            'val_cont_loss_A': loss_A,
            'val_cont_loss_B': loss_B,
            'val_cont_loss_C': loss_C,
            "val_cont_loss": cont_loss,
            "val_pretext_loss": pretext_loss
        }

    def validation_epoch_end(self, outputs):
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_cont_loss_A = torch.stack([x["val_cont_loss_A"] for x in outputs]).mean()
        avg_cont_loss_B = torch.stack([x["val_cont_loss_B"] for x in outputs]).mean()
        avg_cont_loss_C = torch.stack([x["val_cont_loss_C"] for x in outputs]).mean()
        avg_cont_loss = torch.stack([x["val_cont_loss"] for x in outputs]).mean()
        avg_pretext_loss = torch.stack([x["val_pretext_loss"] for x in outputs]).mean()
        self.log("val_recon_loss", avg_recon_loss)
        self.log("val_cont_loss_A", avg_cont_loss_A)
        self.log("val_cont_loss_B", avg_cont_loss_B)
        self.log("val_cont_loss_C", avg_cont_loss_C)
        self.log("val_cont_loss", avg_cont_loss)
        self.log("val_pretext_loss", avg_pretext_loss)
    

class Classifier(pl.LightningModule):
    def __init__(self, ae_model_path, class_weights, num_classes, latent_size, cl_lr, cl_weight_decay, cl_momentum, cl_drop_p, cl_loss, **config):
        super(Classifier, self).__init__()
        self.input_size = latent_size
        self.cl_drop_p = cl_drop_p
        self.num_classes = num_classes
        self.cl_lr = cl_lr
        self.cl_weight_decay = cl_weight_decay
        self.cl_momentum = cl_momentum
        self.class_weights = class_weights
        self.cl_loss = cl_loss
        self.wbce = weighted_binary_cross_entropy
        self.criterion = nn.CrossEntropyLoss()
        self.feature_extractor = AutoEncoder.load_from_checkpoint(ae_model_path)
        self.classifier = ClassifierNet(num_classes, latent_size, dropout_p=cl_drop_p)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--cl_drop_p", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--cl_lr", type=float, default=1e-3)
        parser.add_argument("--cl_weight_decay", type=float, default=0.0001)
        parser.add_argument("--cl_momentum", type=float, default=0.9)
        parser.add_argument("--cl_loss", type=str, default="wbce", help="Loss function to use. Options: wbce, bce")
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
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_pred, y, self.class_weights)
        elif self.cl_loss == "bce":
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
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_pred, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_pred, y)
        y = y.long()
        return {
            "val_down_loss": down_loss,
            "y": y,
            "y_pred": y_pred
        }
    
    def test_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('test_accuracy', summary='max')
        x_A, x_B, x_C, y = batch
        y_pred = self.forward((x_A, x_B, x_C))
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_pred, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_pred, y)
        y = y.long()
        return {
            "test_down_loss": down_loss,
            "y": y,
            "y_pred": y_pred
        }
    
    def validation_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["val_down_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs])
        y_max = torch.argmax(y, dim=1)
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        acc = accuracy(y_pred, y)
        auc = auroc(y_pred, y_max, num_classes=self.num_classes)
        # prec = precision(y_pred, y, num_classes=self.num_classes)
        # rec = recall(y_pred, y, num_classes=self.num_classes)
        # f1 = f1_score(y_pred, y_max, num_classes=self.num_classes)
        y = y.detach().cpu().numpy()
        y_pred = (y_pred.detach().cpu().numpy() > 0.5)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        self.log("val_down_loss", avg_down_loss)
        self.log("val_accuracy", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1", f1)
        self.log("val_auc", auc)
    
    def test_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["test_down_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs])
        y_max = torch.argmax(y, dim=1)
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        acc = accuracy(y_pred, y)
        auc = auroc(y_pred, y_max, num_classes=self.num_classes)
        y = y.detach().cpu().numpy()
        y_pred = (y_pred.detach().cpu().numpy() > 0.5)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        self.log("test_down_loss", avg_down_loss)
        self.log("test_accuracy", acc)
        self.log("test_precision", prec)
        self.log("test_recall", rec)
        self.log("test_f1", f1)
        self.log("test_auc", auc)