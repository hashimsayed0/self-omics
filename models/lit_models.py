from pickletools import optimize
from pl_bolts import optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .networks import SimCLRProjectionHead, CLIPProjectionHead, AENet, ClassifierNet, VAENet
from .losses import SimCLR_Loss, weighted_binary_cross_entropy, CLIPLoss, BarlowTwinsLoss
from .optimizers import LARS
from torchmetrics.functional import f1_score, auroc, precision, recall, accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn as sk
from pytorch_metric_learning import losses
import wandb
import numpy as np

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, ae_net, ae_weight_kl, latent_size, projection_size, ae_lr, ae_weight_decay, ae_momentum, ae_drop_p, cont_loss, cont_loss_temp, cont_loss_lambda, ae_optimizer, ae_use_lrscheduler, cont_loss_weight, split_B, mask_B, num_mask_B, masking_method, batch_size, ae_dim_1B, ae_dim_2B, ae_dim_1A, ae_dim_2A, ae_dim_1C, ae_dim_2C, **config):
        super(AutoEncoder, self).__init__()
        self.input_size_A = input_size_A
        self.input_size_B = input_size_B
        self.input_size_C = input_size_C
        self.ae_net = ae_net
        self.ae_weight_kl = ae_weight_kl
        self.latent_size = latent_size
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay
        self.ae_momentum = ae_momentum
        self.ae_drop_p = ae_drop_p
        self.cont_loss_weight = cont_loss_weight
        self.cont_loss_temp = cont_loss_temp
        self.cont_loss_lambda = cont_loss_lambda
        self.ae_optimizer = ae_optimizer
        self.ae_use_lrscheduler = ae_use_lrscheduler
        self.cont_loss = cont_loss
        self.split_B = split_B
        self.mask_B = mask_B
        self.num_mask_B = num_mask_B
        self.masking_method = masking_method
        
        if self.ae_net == "ae":
            self.net = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        elif self.ae_net == "vae":
            self.net = VAENet((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        
        if cont_loss != "none":
            if cont_loss == "simclr":
                # self.cont_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=projection_size)
                self.projector = SimCLRProjectionHead(latent_size, latent_size, projection_size)
                self.cont_criterion = losses.NTXentLoss(temperature=cont_loss_temp)
            elif cont_loss == "clip":
                self.projector = CLIPProjectionHead(latent_size, projection_size, ae_drop_p)
                self.cont_criterion = CLIPLoss(temperature = cont_loss_temp, latent_size = latent_size, proj_size = projection_size)
            elif cont_loss == "barlowtwins":
                self.cont_criterion = BarlowTwinsLoss(lambd=cont_loss_lambda, latent_size=latent_size, proj_size=projection_size)
        
        if mask_B:
            self.mask_B_ids = np.random.randint(0, len(input_size_B), size=num_mask_B)
        
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument("--ae_net", type=str, default="vae",
                            help="AutoEncoder network architecture, options: [ae, vae]")
        parser.add_argument("--ae_weight_kl", type=float, default=0.01,
                            help="Weight for KL loss if vae is used")
        parser.add_argument("--latent_size", type=int, default=512)
        parser.add_argument("--projection_size", type=int, default=256)
        parser.add_argument("--ae_lr", type=float, default=1e-3)
        parser.add_argument("--ae_weight_decay", type=float, default=0.0001)
        parser.add_argument("--ae_momentum", type=float, default=0.9)
        parser.add_argument("--ae_drop_p", type=float, default=0.2)
        parser.add_argument("--cont_loss", type=str, default="simclr", help="contrastive loss to use, options: none, simclr, clip, barlowtwins")
        parser.add_argument("--cont_loss_temp", type=float, default=0.5)
        parser.add_argument("--cont_loss_lambda", type=float, default=0.0051, help="for barlowtwins")
        parser.add_argument("--cont_loss_weight", type=float, default=0.2)
        parser.add_argument("--ae_optimizer", type=str, default="adam", help="optimizer to use, options: adam, lars")
        parser.add_argument("--ae_use_lrscheduler", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--ae_dim_1B", type=int, default=16)
        parser.add_argument("--ae_dim_2B", type=int, default=32)
        parser.add_argument("--ae_dim_1A", type=int, default=32)
        parser.add_argument("--ae_dim_2A", type=int, default=32)
        parser.add_argument("--ae_dim_1C", type=int, default=32)
        parser.add_argument("--ae_dim_2C", type=int, default=32)
        parser.add_argument("--load_pretrained_ae", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--pretrained_ae_path", type=str, default="")
        parser.add_argument('--mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, num_mask_B chromosomes of B are masked')
        parser.add_argument('--num_mask_B', type=int, default=0,
                                help='number of chromosomes of B to mask')
        parser.add_argument('--masking_method', type=str, default='zero',
                                help='method to mask data, can be "zero" or "noise"')
        return parent_parser

    def forward(self, x):
        if self.ae_net == "vae":
            return self.net(x)
        elif self.ae_net == "ae":
            return self.net.encode(x)

    def configure_optimizers(self):
        if self.ae_optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay)
        if self.ae_optimizer == "lars":
            optimizer = LARS(self.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay)
        if self.ae_use_lrscheduler:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=500,
                                                            eta_min=self.hparams.ae_lr/50)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        if self.ae_net == 'ae':
            x_A, x_B, x_C, _ = batch
            h_A, h_B, h_C = self.net.encode((x_A, x_B, x_C))
            x_A_recon, x_B_recon, x_C_recon = self.net.decode((h_A, h_B, h_C))
            if self.split_B:
                x_B_recon_loss = []
                for i in range(len(x_B)):
                    x_B_recon_loss.append(F.mse_loss(x_B_recon[i], x_B[i]))
                recon_loss = F.mse_loss(x_A_recon, x_A) + sum(x_B_recon_loss) + F.mse_loss(x_C_recon, x_C)        
            self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        
        elif self.ae_net == 'vae':
            x_A, x_B, x_C, _ = batch
            if self.mask_B:
                x_B_masked = []
                for i in range(len(x_B)):
                    x_B_masked.append(x_B[i])
                    if i in self.mask_B_ids:
                        if self.masking_method == 'zero':
                            x_B_masked[-1] = torch.zeros_like(x_B_masked[-1])
                        elif self.masking_method == 'noise':
                            x_B_masked[-1] = x_B_masked[-1] + torch.randn_like(x_B_masked[-1])

            z, recon_x, mean, log_var = self.net((x_A, x_B_masked, x_C))
            x_B_recon_loss = []
            for i in range(len(x_B_masked)):
                x_B_recon_loss.append(F.mse_loss(recon_x[1][i], x_B[i]))
            recon_B_loss = sum(x_B_recon_loss)
            recon_loss_all = F.mse_loss(recon_x[0], x_A) + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_B_loss + kl_loss
            self.log("train_recon_all_loss", recon_loss_all, on_step=False, on_epoch=True)
            self.log("train_kl_loss", kl_loss, on_step=False, on_epoch=True)
            self.log("train_recon_B_kl_loss", recon_loss, on_step=False, on_epoch=True)
            self.log("train_recon_B_loss", recon_B_loss, on_step=False, on_epoch=True)

        if self.cont_loss != "none":
            if self.cont_loss == "clip":
                loss_A_B, loss_A1, loss_B1 = self.cont_criterion(h_A, h_B)
                loss_B_C, loss_B2, loss_C1 = self.cont_criterion(h_B, h_C)
                loss_C_A, loss_C2, loss_A2 = self.cont_criterion(h_C, h_A)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_A = (loss_A1 + loss_A2) / 2
                loss_B = (loss_B1 + loss_B2) / 2
                loss_C = (loss_C1 + loss_C2) / 2
                self.log("train_cont_loss_A", loss_A, on_step=False, on_epoch=True)
                self.log("train_cont_loss_B", loss_B, on_step=False, on_epoch=True)
                self.log("train_cont_loss_C", loss_C, on_step=False, on_epoch=True)

            elif self.cont_loss == "barlowtwins":
                loss_A_B, loss_on_diag1, loss_off_diag1 = self.cont_criterion(h_A, h_B)
                loss_B_C, loss_on_diag2, loss_off_diag2 = self.cont_criterion(h_B, h_C)
                loss_C_A, loss_on_diag3, loss_off_diag3 = self.cont_criterion(h_C, h_A)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_on_diag = (loss_on_diag1 + loss_on_diag2 + loss_on_diag3) / 3
                loss_off_diag = (loss_off_diag1 + loss_off_diag2 + loss_off_diag3) / 3
                self.log("train_cont_loss_on_diag", loss_on_diag, on_step=False, on_epoch=True)
                self.log("train_cont_loss_off_diag", loss_off_diag, on_step=False, on_epoch=True)
            
            elif self.cont_loss == "simclr":
                z_A = self.projector(h_A)
                z_B = self.projector(h_B)
                z_C = self.projector(h_C)
                z_AB = torch.cat((z_A, z_B), dim=0)
                z_BC = torch.cat((z_B, z_C), dim=0)
                z_CA = torch.cat((z_C, z_A), dim=0)
                labels = torch.arange(z_A.shape[0]).repeat(2)
                loss_A_B, loss_num1, loss_den1 = self.cont_criterion(z_AB, labels)
                loss_B_C, loss_num2, loss_den2 = self.cont_criterion(z_BC, labels)
                loss_C_A, loss_num3, loss_den3 = self.cont_criterion(z_CA, labels)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_num = (loss_num1 + loss_num2 + loss_num3) / 3
                loss_den = (loss_den1 + loss_den2 + loss_den3) / 3
                self.log("train_cont_loss_num", loss_num, on_step=False, on_epoch=True)
                self.log("train_cont_loss_den", loss_den, on_step=False, on_epoch=True)

            pretext_loss = recon_loss + self.cont_loss_weight * cont_loss
            self.log("train_cont_loss", cont_loss, on_step=False, on_epoch=True)
            self.log("train_pretext_loss", pretext_loss, on_step=False, on_epoch=True)
            return pretext_loss
        else:
            return recon_loss

    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_pretext_loss', summary='min')
            wandb.define_metric('val_recon_B_kl_loss', summary='min')
        if self.ae_net == 'ae':
            x_A, x_B, x_C, _ = batch
            h_A, h_B, h_C = self.net.encode((x_A, x_B, x_C))
            x_A_recon, x_B_recon, x_C_recon = self.net.decode((h_A, h_B, h_C))
            if self.split_B:
                x_B_recon_loss = []
                for i in range(len(x_B)):
                    x_B_recon_loss.append(F.mse_loss(x_B_recon[i], x_B[i]))
                recon_loss = F.mse_loss(x_A_recon, x_A) + sum(x_B_recon_loss) + F.mse_loss(x_C_recon, x_C)    
            logs = {'val_recon_loss': recon_loss}

        elif self.ae_net == 'vae':
            x_A, x_B, x_C, _ = batch
            if self.mask_B:
                x_B_masked = []
                for i in range(len(x_B)):
                    x_B_masked.append(x_B[i])
                    if i in self.mask_B_ids:
                        if self.masking_method == 'zero':
                            x_B_masked[-1] = torch.zeros_like(x_B_masked[-1])
                        elif self.masking_method == 'noise':
                            x_B_masked[-1] = x_B_masked[-1] + torch.randn_like(x_B_masked[-1])

            z, recon_x, mean, log_var = self.net((x_A, x_B_masked, x_C))
            x_B_recon_loss = []
            for i in range(len(x_B_masked)):
                x_B_recon_loss.append(F.mse_loss(recon_x[1][i], x_B[i]))
            recon_B_loss = sum(x_B_recon_loss)
            recon_loss_all = F.mse_loss(recon_x[0], x_A) + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_B_loss + kl_loss
            logs = {
                'val_recon_all_loss': recon_loss_all, 
                'val_kl_loss': kl_loss,
                'val_recon_B_kl_loss': recon_loss,
                'val_recon_B_loss': recon_B_loss
            }
        
        if self.cont_loss != "none":
            if self.cont_loss == "clip":
                loss_A_B, loss_A1, loss_B1 = self.cont_criterion(h_A, h_B)
                loss_B_C, loss_B2, loss_C1 = self.cont_criterion(h_B, h_C)
                loss_C_A, loss_C2, loss_A2 = self.cont_criterion(h_C, h_A)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_A = (loss_A1 + loss_A2) / 2
                loss_B = (loss_B1 + loss_B2) / 2
                loss_C = (loss_C1 + loss_C2) / 2
                logs['val_cont_loss_A'] = loss_A
                logs['val_cont_loss_B'] = loss_B
                logs['val_cont_loss_C'] = loss_C

            elif self.cont_loss == "barlowtwins":
                loss_A_B, loss_on_diag1, loss_off_diag1 = self.cont_criterion(h_A, h_B)
                loss_B_C, loss_on_diag2, loss_off_diag2 = self.cont_criterion(h_B, h_C)
                loss_C_A, loss_on_diag3, loss_off_diag3 = self.cont_criterion(h_C, h_A)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_on_diag = (loss_on_diag1 + loss_on_diag2 + loss_on_diag3) / 3
                loss_off_diag = (loss_off_diag1 + loss_off_diag2 + loss_off_diag3) / 3
                logs['val_cont_loss_on_diag'] = loss_on_diag
                logs['val_cont_loss_off_diag'] = loss_off_diag
            
            elif self.cont_loss == "simclr":
                z_A = self.projector(h_A)
                z_B = self.projector(h_B)
                z_C = self.projector(h_C)
                z_AB = torch.cat((z_A, z_B), dim=0)
                z_BC = torch.cat((z_B, z_C), dim=0)
                z_CA = torch.cat((z_C, z_A), dim=0)
                labels = torch.arange(z_A.shape[0]).repeat(2)
                loss_A_B, loss_num1, loss_den1 = self.cont_criterion(z_AB, labels)
                loss_B_C, loss_num2, loss_den2 = self.cont_criterion(z_BC, labels)
                loss_C_A, loss_num3, loss_den3 = self.cont_criterion(z_CA, labels)
                cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
                loss_num = (loss_num1 + loss_num2 + loss_num3) / 3
                loss_den = (loss_den1 + loss_den2 + loss_den3) / 3
                logs['val_cont_loss_num'] = loss_num
                logs['val_cont_loss_den'] = loss_den

            pretext_loss = recon_loss + self.cont_loss_weight * cont_loss
            logs['val_cont_loss'] = cont_loss
            logs['val_pretext_loss'] = pretext_loss
        
        return logs

    def validation_epoch_end(self, outputs):
        if self.ae_net == 'ae':
            avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
            self.log("val_recon_loss", avg_recon_loss)
        elif self.ae_net == 'vae':
            avg_recon_all_loss = torch.stack([x["val_recon_all_loss"] for x in outputs]).mean()
            avg_kl_loss = torch.stack([x["val_kl_loss"] for x in outputs]).mean()
            avg_recon_B_kl_loss = torch.stack([x["val_recon_B_kl_loss"] for x in outputs]).mean()
            avg_recon_B_loss = torch.stack([x["val_recon_B_loss"] for x in outputs]).mean()
            self.log("val_recon_all_loss", avg_recon_all_loss)
            self.log("val_kl_loss", avg_kl_loss)
            self.log("val_recon_B_kl_loss", avg_recon_B_kl_loss)
            self.log("val_recon_B_loss", avg_recon_B_loss)
        
        if self.cont_loss != "none":
            if self.cont_loss == "clip":
                avg_cont_loss_A = torch.stack([x["val_cont_loss_A"] for x in outputs]).mean()
                avg_cont_loss_B = torch.stack([x["val_cont_loss_B"] for x in outputs]).mean()
                avg_cont_loss_C = torch.stack([x["val_cont_loss_C"] for x in outputs]).mean()
                self.log("val_cont_loss_A", avg_cont_loss_A)
                self.log("val_cont_loss_B", avg_cont_loss_B)
                self.log("val_cont_loss_C", avg_cont_loss_C)
            elif self.cont_loss == "barlowtwins":
                avg_cont_loss_on_diag = torch.stack([x["val_cont_loss_on_diag"] for x in outputs]).mean()
                avg_cont_loss_off_diag = torch.stack([x["val_cont_loss_off_diag"] for x in outputs]).mean()
                self.log("val_cont_loss_on_diag", avg_cont_loss_on_diag)
                self.log("val_cont_loss_off_diag", avg_cont_loss_off_diag)
            elif self.cont_loss == "simclr":
                avg_cont_loss_num = torch.stack([x["val_cont_loss_num"] for x in outputs]).mean()
                avg_cont_loss_den = torch.stack([x["val_cont_loss_den"] for x in outputs]).mean()
                self.log("val_cont_loss_num", avg_cont_loss_num)
                self.log("val_cont_loss_den", avg_cont_loss_den)
            avg_cont_loss = torch.stack([x["val_cont_loss"] for x in outputs]).mean()
            avg_pretext_loss = torch.stack([x["val_pretext_loss"] for x in outputs]).mean()
            self.log("val_cont_loss", avg_cont_loss)
            self.log("val_pretext_loss", avg_pretext_loss)

class Classifier(pl.LightningModule):
    def __init__(self, ae_model_path, class_weights, num_classes, ae_net, latent_size, cl_lr, cl_weight_decay, cl_beta1, cl_drop_p, cl_loss, cl_lr_policy, cl_epoch_num_decay, cl_decay_step_size, max_epochs, **config):
        super(Classifier, self).__init__()
        self.input_size = latent_size
        self.cl_drop_p = cl_drop_p
        self.num_classes = num_classes
        self.cl_lr = cl_lr
        self.cl_weight_decay = cl_weight_decay
        self.cl_beta1 = cl_beta1
        self.cl_lr_policy = cl_lr_policy
        self.cl_epoch_num_decay = cl_epoch_num_decay
        self.cl_decay_step_size = cl_decay_step_size
        self.class_weights = class_weights
        self.cl_loss = cl_loss
        self.ae_net = ae_net
        self.cl_max_epochs = max_epochs
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
        parser.add_argument("--cl_lr", type=float, default=1e-4)
        parser.add_argument("--cl_weight_decay", type=float, default=1e-4)
        parser.add_argument("--cl_beta1", type=float, default=0.5)
        parser.add_argument('--cl_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine]')
        parser.add_argument('--cl_epoch_num_decay', type=int, default=50,
                            help='Number of epoch to linearly decay learning rate to zero (lr_policy == linear)')
        parser.add_argument('--cl_decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch (lr_policy == step)')
        parser.add_argument("--cl_loss", type=str, default="wbce", help="Loss function to use. Options: wbce, bce")
        return parent_parser

    def forward(self, x):
        if self.ae_net == 'ae':
            h_A, h_B, h_C = self.feature_extractor(x)
            h = torch.mean(torch.stack([h_A, h_B, h_C]), axis=0)
        elif self.ae_net == 'vae':
            h, _, _, _ = self.feature_extractor(x)
        return self.classifier(h)

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.cl_lr, weight_decay=self.cl_weight_decay, betas=(self.cl_beta1, 0.999))
        if self.cl_lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_lambda = 1.0 - max(0, epoch - self.cl_max_epochs + self.cl_epoch_num_decay) / float(self.cl_epoch_num_decay + 1)
                return lr_lambda
            # lr_scheduler is imported from torch.optim
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.cl_lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.cl_decay_step_size, gamma=0.1)
        elif self.cl_lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif self.cl_lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cl_max_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x_A, x_B, x_C, y = batch
        y_out = self.forward((x_A, x_B, x_C))
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_out, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_out, y)
        y_true = y.long()
        y_prob = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_prob, 1)

        return {
            "loss": down_loss,
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach()
        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_down_loss", avg_loss)

        y_true_binary = torch.cat([x["y_true"] for x in outputs]).cpu().numpy()
        y_true = np.argmax(y_true_binary, axis=1)
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().numpy()
        y_prob = torch.cat([x["y_prob"] for x in outputs]).cpu().numpy()
        
        accuracy = sk.metrics.accuracy_score(y_true, y_pred)
        precision = sk.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = sk.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = sk.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auc = sk.metrics.roc_auc_score(y_true_binary, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            auc = -1
            print('ValueError: Train ROC AUC score is not defined in this case.')
        
        self.log("train_accuracy", accuracy)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
        self.log("train_f1", f1)
        self.log("train_auc", auc)
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_accuracy', summary='max')
        x_A, x_B, x_C, y = batch
        y_out = self.forward((x_A, x_B, x_C))
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_out, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_out, y)
        y_true = y.long()
        y_prob = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_prob, 1)
        return {
            "val_down_loss": down_loss,
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach()
        }

    def validation_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["val_down_loss"] for x in outputs]).mean()
        self.log("val_down_loss", avg_down_loss)

        y_true_binary = torch.cat([x["y_true"] for x in outputs]).cpu().numpy()
        y_true = np.argmax(y_true_binary, axis=1)
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().numpy()
        y_prob = torch.cat([x["y_prob"] for x in outputs]).cpu().numpy()
        
        accuracy = sk.metrics.accuracy_score(y_true, y_pred)
        precision = sk.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = sk.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = sk.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auc = sk.metrics.roc_auc_score(y_true_binary, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            auc = -1
            print('ValueError: Validation ROC AUC score is not defined in this case.')

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_auc", auc)
    
    def test_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('test_accuracy', summary='max')
        x_A, x_B, x_C, y = batch
        y_out = self.forward((x_A, x_B, x_C))
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_out, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_out, y)
        y_true = y.long()
        y_prob = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_prob, 1)
        return {
            "test_down_loss": down_loss,
            "y_true": y_true.detach(),
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach()
        }
    
    def test_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["test_down_loss"] for x in outputs]).mean()
        self.log("test_down_loss", avg_down_loss)

        y_true_binary = torch.cat([x["y_true"] for x in outputs]).cpu().numpy()
        y_true = np.argmax(y_true_binary, axis=1)
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().numpy()
        y_prob = torch.cat([x["y_prob"] for x in outputs]).cpu().numpy()
        
        accuracy = sk.metrics.accuracy_score(y_true, y_pred)
        precision = sk.metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = sk.metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = sk.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auc = sk.metrics.roc_auc_score(y_true_binary, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            auc = -1
            print('ValueError: Validation ROC AUC score is not defined in this case.')

        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_auc", auc)