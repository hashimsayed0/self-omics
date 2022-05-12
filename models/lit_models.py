from email.policy import default
from pickletools import optimize
from random import sample
from pl_bolts import optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .networks import SimCLRProjectionHead, CLIPProjectionHead, AESepB, AESepAB, ClassifierNet, VAESepB, VAESepAB, SurvivalNet
from .losses import SimCLR_Loss, weighted_binary_cross_entropy, CLIPLoss, BarlowTwinsLoss, MTLR_survival_loss
from .optimizers import LARS
from torchmetrics.functional import f1_score, auroc, precision, recall, accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import sklearn as sk
from pytorch_metric_learning import losses
import wandb
import numpy as np
import utils.metrics as metrics
import pandas as pd
import os

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, ae_net, ae_weight_kl, latent_size, projection_size, ae_lr, ae_weight_decay, ae_momentum, ae_drop_p, ae_beta1, ae_lr_policy, ae_epoch_num_decay, ae_decay_step_size, max_epochs, cont_loss_criterion, cont_loss_temp, cont_loss_lambda, ae_optimizer, ae_use_lrscheduler, cont_loss_weight, split_A, split_B, mask_A, mask_B, num_mask_A, num_mask_B, masking_method, recon_all, batch_size, ae_dim_1B, ae_dim_2B, ae_dim_1A, ae_dim_2A, ae_dim_1C, ae_dim_2C, **config):
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
        self.ae_beta1 = ae_beta1
        self.ae_lr_policy = ae_lr_policy
        self.ae_epoch_num_decay = ae_epoch_num_decay
        self.ae_decay_step_size = ae_decay_step_size
        self.ae_max_epochs = max_epochs
        self.cont_loss_weight = cont_loss_weight
        self.cont_loss_temp = cont_loss_temp
        self.cont_loss_lambda = cont_loss_lambda
        self.add_distance_loss = config['add_distance_loss']
        self.distance_loss_weight = config['distance_loss_weight']
        self.ae_optimizer = ae_optimizer
        self.ae_use_lrscheduler = ae_use_lrscheduler
        self.cont_loss_criterion = cont_loss_criterion
        self.cont_loss = config['cont_loss']
        self.split_A = split_A
        self.mask_A = mask_A
        self.num_mask_A = num_mask_A
        self.split_B = split_B
        self.mask_B = mask_B
        self.num_mask_B = num_mask_B
        self.masking_method = masking_method
        self.choose_masking_method_every_epoch = config['choose_masking_method_every_epoch']
        self.recon_all = recon_all
        self.use_one_decoder = config['use_one_decoder']
        self.concat_latent_for_decoder = config['concat_latent_for_decoder']
        
        if self.ae_net == "ae":
            if self.split_A and self.split_B:
                self.net = AESepAB((input_size_A, input_size_B, input_size_C), latent_size, self.use_one_decoder, self.concat_latent_for_decoder, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
            elif self.split_B:
                self.net = AESepB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        elif self.ae_net == "vae":
            if self.split_A and self.split_B:
                self.net = VAESepAB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
            elif self.split_B:
                self.net = VAESepB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        
        if self.cont_loss != "none":
            if self.cont_loss == 'both':
                self.cont_loss_pairs = ['patient', 'type']
            else:
                self.cont_loss_pairs = [self.cont_loss]

            self.projection_size = latent_size // 2
            if cont_loss_criterion == "simclr":
                # self.cont_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
                self.projector = SimCLRProjectionHead(latent_size, latent_size, self.projection_size)
                self.cont_criterion = losses.NTXentLoss(temperature=cont_loss_temp)
            elif cont_loss_criterion == "clip":
                self.projector = CLIPProjectionHead(latent_size, self.projection_size, ae_drop_p)
                self.cont_criterion = CLIPLoss(temperature = cont_loss_temp, latent_size = latent_size, proj_size = self.projection_size)
            elif cont_loss_criterion == "barlowtwins":
                self.cont_criterion = BarlowTwinsLoss(lambd=cont_loss_lambda, latent_size=latent_size, proj_size=self.projection_size)
        
        if self.add_distance_loss:
            self.dist_loss = nn.MSELoss()
        if self.mask_B:
            self.mask_B_ids = np.random.randint(0, len(self.input_size_B), size=self.num_mask_B)
        if self.mask_A:
            self.mask_A_ids = np.random.randint(0, len(self.input_size_A), size=self.num_mask_A)
        
        self.change_ch_to_mask_every_epoch = config['change_ch_to_mask_every_epoch']

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
        parser.add_argument("--ae_weight_decay", type=float, default=1e-4)
        parser.add_argument("--ae_momentum", type=float, default=0.9)
        parser.add_argument("--ae_drop_p", type=float, default=0.2)
        parser.add_argument("--cont_loss", type=str, default="none",
                            help="Contrastive loss, options: [none, patient, type (cancer type), both]")
        parser.add_argument("--cont_loss_criterion", type=str, default="barlowtwins", help="contrastive loss to use, options: none, simclr, clip, barlowtwins")
        parser.add_argument("--cont_loss_temp", type=float, default=0.1)
        parser.add_argument("--cont_loss_lambda", type=float, default=0.0051, help="for barlowtwins")
        parser.add_argument("--cont_loss_weight", type=float, default=0.5)
        parser.add_argument("--add_distance_loss", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--distance_loss_weight", type=float, default=0.5)
        parser.add_argument("--ae_optimizer", type=str, default="adam", help="optimizer to use, options: adam, lars")
        parser.add_argument("--ae_use_lrscheduler", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--ae_beta1", type=float, default=0.5)
        parser.add_argument('--ae_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine]')
        parser.add_argument('--ae_epoch_num_decay', type=int, default=50,
                            help='Number of epoch to linearly decay learning rate to zero (lr_policy == linear)')
        parser.add_argument('--ae_decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch (lr_policy == step)')
        parser.add_argument("--ae_dim_1B", type=int, default=128)
        parser.add_argument("--ae_dim_2B", type=int, default=1024)
        parser.add_argument("--ae_dim_1A", type=int, default=2048)
        parser.add_argument("--ae_dim_2A", type=int, default=1024)
        parser.add_argument("--ae_dim_1C", type=int, default=1024)
        parser.add_argument("--ae_dim_2C", type=int, default=1024)
        parser.add_argument("--load_pretrained_ae", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--pretrained_ae_path", type=str, default="")
        parser.add_argument('--mask_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, num_mask_A chromosomes of A are masked')
        parser.add_argument('--num_mask_A', type=int, default=0,
                                help='number of chromosomes of A to mask')
        parser.add_argument('--mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, num_mask_B chromosomes of B are masked')
        parser.add_argument('--num_mask_B', type=int, default=0,
                                help='number of chromosomes of B to mask')
        parser.add_argument('--masking_method', type=str, default='zero',
                                help='method to mask data, can be "zero", "gaussian_noise", or "swap_noise"')
        parser.add_argument('--choose_masking_method_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the masking method is chosen randomly each epoch and "masking_method" argument is ignored')
        parser.add_argument('--change_ch_to_mask_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the chromosomes to mask are changed each epoch')
        parser.add_argument('--recon_all', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, modalities A, B and C will be reconstructed with higher weightage to masked modality, else, only the masked modality will be reconstructed')
        parser.add_argument('--use_one_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, only one decoder is used to reconstruct all modalities')
        parser.add_argument('--concat_latent_for_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, latent vectors from A, B and C are concatenated before being fed into the decoder')
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
        return optimizer
    
    def train(self, mode=True):
        super().train(mode)
        self.mode = "train"
    
    def eval(self):
        super().eval()
        self.mode = "val"
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.change_ch_to_mask_every_epoch:
            if self.mask_B:
                self.mask_B_ids = np.random.randint(0, len(self.input_size_B), size=self.num_mask_B)
            if self.mask_A:
                self.mask_A_ids = np.random.randint(0, len(self.input_size_A), size=self.num_mask_A)
        if self.choose_masking_method_every_epoch:
            self.masking_method = np.random.choice(['zero', 'gaussian_noise', 'swap_noise'])

    def mask_x(self, x, mask_ids):
        x_masked = []
        for i in range(len(x)):
            x_masked.append(x[i])
            if i in mask_ids:
                if self.masking_method == 'zero':
                    x_masked[-1] = torch.zeros_like(x_masked[-1])
                elif self.masking_method == 'gaussian_noise':
                    x_masked[-1] = x_masked[-1] + torch.randn_like(x_masked[-1])
                elif self.masking_method == 'swap_noise':
                    for j in range(x_masked[-1].shape[1]):
                        x_masked[-1][:, j] = x_masked[-1][torch.randperm(x_masked[-1].shape[0]), j]
        return x_masked
    
    def sum_subset_losses(self, x_recon, x):
        x_recon_loss = []
        for i in range(len(x)):
            x_recon_loss.append(F.mse_loss(x_recon[i], x[i]))
        return sum(x_recon_loss)

    def ae_step(self, batch):
        logs = {}
        x_A, x_B, x_C = batch['x']
        if self.mask_A:
            x_A_masked = self.mask_x(x_A, self.mask_A_ids)
            h_A, h_B, h_C = self.net.encode((x_A_masked, x_B, x_C))
            x_A_recon, x_B_recon, x_C_recon = self.net.decode((h_A, h_B, h_C))
            recon_A_loss = self.sum_subset_losses(x_A_recon, x_A)
            if self.split_B:
                recon_B_loss = self.sum_subset_losses(x_B_recon, x_B)
            
            recon_loss_all = 0.5 * recon_A_loss + 0.25 * recon_B_loss + 0.25 * F.mse_loss(x_C_recon, x_C)
            recon_loss = recon_loss_all

            logs['{}_recon_all_loss'.format(self.mode)] = recon_loss_all
            logs['{}_recon_A_loss'.format(self.mode)] = recon_A_loss
        
        elif self.mask_B:
            x_B_masked = self.mask_x(x_B, self.mask_B_ids)
            h_A, h_B, h_C = self.net.encode((x_A, x_B_masked, x_C))
            x_A_recon, x_B_recon, x_C_recon = self.net.decode((h_A, h_B, h_C))
            recon_B_loss = self.sum_subset_losses(x_B_recon, x_B)
            
            recon_loss_all = 0.25 * F.mse_loss(x_A_recon, x_A) + 0.5 * recon_B_loss + 0.25 * F.mse_loss(x_C_recon, x_C)
            recon_loss = recon_loss_all

            logs['{}_recon_all_loss'.format(self.mode)] = recon_loss_all
            logs['{}_recon_B_loss'.format(self.mode)] = recon_B_loss
        
        else:
            h_A, h_B, h_C = self.net.encode((x_A, x_B, x_C))
            x_A_recon, x_B_recon, x_C_recon = self.net.decode((h_A, h_B, h_C))
            if self.split_B:
                recon_B_loss = self.sum_subset_losses(x_B_recon, x_B)
            if self.split_A:
                recon_A_loss = self.sum_subset_losses(x_A_recon, x_A)
            else:
                recon_A_loss = F.mse_loss(x_A_recon, x_A)
            recon_loss = recon_A_loss + recon_B_loss + F.mse_loss(x_C_recon, x_C) 
            logs["{}_recon_loss".format(self.mode)] = recon_loss
        
        return logs, (h_A, h_B, h_C), recon_loss
    
    def vae_step(self, batch):
        logs = {}
        x_A, x_B, x_C = batch['x']
        if self.mask_B:
            x_B_masked = self.mask_x(x_B, self.mask_B_ids)
            z, recon_x, mean, log_var = self.net((x_A, x_B_masked, x_C))
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = F.mse_loss(recon_x[0], x_A) + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_B_loss + kl_loss
            logs['{}_recon_B_loss'.format(self.mode)] = recon_B_loss
            logs['{}_recon_B_kl_loss'.format(self.mode)] = recon_loss
        
        elif self.mask_A:
            x_A_masked = self.mask_x(x_A, self.mask_A_ids)
            h, recon_x, mean, log_var = self.net((x_A_masked, x_B, x_C))
            recon_A_loss = self.sum_subset_losses(recon_x[0], x_A)
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = recon_A_loss + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_A_loss + kl_loss
            logs['{}_recon_A_loss'.format(self.mode)] = recon_A_loss
            logs['{}_recon_A_kl_loss'.format(self.mode)] = recon_loss

        else:
            h, recon_x, mean, log_var = self.net((x_A, x_B, x_C))
            recon_A_loss = F.mse_loss(recon_x[0], x_A)
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_C_loss = F.mse_loss(recon_x[2], x_C)
            recon_loss_all = recon_A_loss + recon_B_loss + recon_C_loss
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_A_loss + recon_B_loss + recon_C_loss + kl_loss
            logs['{}_recon_all_kl_loss'.format(self.mode)] = recon_loss

            
        logs['{}_recon_all_loss'.format(self.mode)] = recon_loss_all
        logs['{}_kl_loss'.format(self.mode)] = kl_loss

        return logs, h, recon_loss

    def prepare_cont_type_h(self, h, y):
        h_cont_list = []
        labels = torch.argmax(y, dim=1)
        while True:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            unique_labels = unique_labels[counts >= 3]
            if len(unique_labels) < 2:
                break
            h1, h2, h3 = h
            h1_cont = torch.zeros(len(unique_labels), h1.shape[1]).to(self.device)
            h2_cont = torch.zeros(len(unique_labels), h2.shape[1]).to(self.device)
            h3_cont = torch.zeros(len(unique_labels), h3.shape[1]).to(self.device)
            for i, unique_label in enumerate(unique_labels):
                indices = (labels == unique_label).nonzero()[:3]
                h1_cont[i,:] = h1[indices[0],:]
                h2_cont[i,:] = h2[indices[1],:]
                h3_cont[i,:] = h3[indices[2],:]
                new_indices = [idx for idx in range(len(labels)) if idx not in indices]
                labels = labels[new_indices]
            h_cont_list.append((h1_cont, h2_cont, h3_cont))
        return h_cont_list
    
    def cont_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        if self.cont_loss_criterion == "clip":
            loss_A_B, loss_A1, loss_B1 = self.cont_criterion(h_A, h_B)
            loss_B_C, loss_B2, loss_C1 = self.cont_criterion(h_B, h_C)
            loss_C_A, loss_C2, loss_A2 = self.cont_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_A = (loss_A1 + loss_A2) / 2
            loss_B = (loss_B1 + loss_B2) / 2
            loss_C = (loss_C1 + loss_C2) / 2
            logs['{}_cont_{}_loss_A'.format(self.mode, self.cont_pair)] = loss_A
            logs['{}_cont_{}_loss_B'.format(self.mode, self.cont_pair)] = loss_B
            logs['{}_cont_{}_loss_C'.format(self.mode, self.cont_pair)] = loss_C

        elif self.cont_loss_criterion == "barlowtwins":
            loss_A_B, loss_on_diag1, loss_off_diag1 = self.cont_criterion(h_A, h_B)
            loss_B_C, loss_on_diag2, loss_off_diag2 = self.cont_criterion(h_B, h_C)
            loss_C_A, loss_on_diag3, loss_off_diag3 = self.cont_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_on_diag = (loss_on_diag1 + loss_on_diag2 + loss_on_diag3) / 3
            loss_off_diag = (loss_off_diag1 + loss_off_diag2 + loss_off_diag3) / 3
            logs['{}_cont_{}_loss_on_diag'.format(self.mode, self.cont_pair)] = loss_on_diag
            logs['{}_cont_{}_loss_off_diag'.format(self.mode, self.cont_pair)] = loss_off_diag
        
        elif self.cont_loss_criterion == "simclr":
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
            logs['{}_cont_{}_loss_num'.format(self.mode, self.cont_pair)] = loss_num
            logs['{}_cont_{}_loss_den'.format(self.mode, self.cont_pair)] = loss_den

        logs['{}_cont_{}_loss'.format(self.mode, self.cont_pair)] = cont_loss
        return logs, cont_loss
        
    def dist_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        dist_loss = self.dist_loss(h_A, h_B) + self.dist_loss(h_B, h_C) + self.dist_loss(h_C, h_A)
        return logs, dist_loss
    
    def training_step(self, batch, batch_idx):
        if self.ae_net == 'ae':
            logs, h, pretext_loss = self.ae_step(batch) 
        elif self.ae_net == 'vae':
            logs, h, pretext_loss = self.vae_step(batch)
        for k, v in logs.items():
            self.log(k, v, on_step=False, on_epoch=True)
        if self.add_distance_loss:
            logs, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.distance_loss_weight
            self.log('{}_dist_loss'.format(self.mode), dist_loss, on_step=False, on_epoch=True)
        if self.cont_loss != "none":
            if 'patient' in self.cont_loss_pairs:
                self.cont_pair = 'patient'
                logs, cont_loss = self.cont_step(h)
                for k, v in logs.items():
                    self.log(k, v, on_step=False, on_epoch=True)
                pretext_loss += self.cont_loss_weight * cont_loss
            if 'type' in self.cont_loss_pairs:
                h_list = self.prepare_cont_type_h(h, batch['y'])
                for i, h_type in enumerate(h_list):
                    self.cont_pair = 'type'
                    logs, cont_loss = self.cont_step(h_type)
                    for k, v in logs.items():
                        self.log(k, v, on_step=False, on_epoch=True)
                    pretext_loss += self.cont_loss_weight * cont_loss
            logs['{}_pretext_loss'.format(self.mode)] = pretext_loss
        return pretext_loss
    
    def validation_step(self, batch, batch_idx):
        # if self.global_step == 0: 
        #     wandb.define_metric('val_pretext_loss', summary='min')
        #     wandb.define_metric('val_recon_loss', summary='min')
        
        if self.ae_net == 'ae':
            logs, h, pretext_loss = self.ae_step(batch) 
        elif self.ae_net == 'vae':
            logs, h, pretext_loss = self.vae_step(batch)
        if self.add_distance_loss:
            dist_logs, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.distance_loss_weight
            logs['{}_dist_loss'.format(self.mode)] = dist_loss
        if self.cont_loss != "none":
            cont_logs = {}
            if 'patient' in self.cont_loss_pairs:
                self.cont_pair = 'patient'
                cont_pair_logs, cont_loss = self.cont_step(h)
                cont_logs.update(cont_pair_logs)
                pretext_loss += self.cont_loss_weight * cont_loss
            if 'type' in self.cont_loss_pairs:
                h_list = self.prepare_cont_type_h(h, batch['y'])
                for i, h_type in enumerate(h_list):
                    self.cont_pair = 'type'
                    cont_pair_logs, cont_loss = self.cont_step(h_type)
                    cont_logs.update(cont_pair_logs)
                    pretext_loss += self.cont_loss_weight * cont_loss
            cont_logs['{}_pretext_loss'.format(self.mode)] = pretext_loss
            return {**logs, **cont_logs}
        else:
            return logs

    def validation_epoch_end(self, outputs):
        for key, value in outputs[0].items():
            avg = torch.stack([x[key] for x in outputs]).mean()
            self.log(key, avg)
        


class DownstreamModel(pl.LightningModule):
    def __init__(self, ae_model_path, class_weights, checkpoint_path, num_classes, ae_net, latent_size, ds_lr, ds_weight_decay, ds_beta1, ds_drop_p, cl_loss, ds_lr_policy, ds_epoch_num_decay, ds_decay_step_size, max_epochs, **config):
        super(DownstreamModel, self).__init__()
        self.input_size = latent_size
        self.ds_drop_p = ds_drop_p
        self.num_classes = num_classes
        self.ds_lr = ds_lr
        self.ds_weight_decay = ds_weight_decay
        self.ds_beta1 = ds_beta1
        self.ds_lr_policy = ds_lr_policy
        self.ds_epoch_num_decay = ds_epoch_num_decay
        self.ds_decay_step_size = ds_decay_step_size
        self.class_weights = class_weights
        self.ae_net = ae_net
        self.ds_max_epochs = max_epochs
        self.ds_task = config['ds_task']
        self.ds_save_latent_testing = config['ds_save_latent_testing']
        self.checkpoint_path = checkpoint_path
        self.ds_mask_A = config['ds_mask_A']
        self.ds_mask_B = config['ds_mask_B']
        self.ds_masking_method = config['ds_masking_method']
        self.feature_extractor = AutoEncoder.load_from_checkpoint(ae_model_path)
        if config['ds_freeze_ae'] == True:
            self.feature_extractor.freeze()
        self.ds_latent_agg_method = config['ds_latent_agg_method']
        if self.ds_latent_agg_method == 'concat':
            latent_size *= 3
        if self.ds_task == 'class':
            self.ds_net = ClassifierNet(num_classes, latent_size, dropout_p=ds_drop_p)
            self.wbce = weighted_binary_cross_entropy
            self.criterion = nn.CrossEntropyLoss()
            self.cl_loss = cl_loss
        elif self.ds_task == 'surv':
            self.time_num = config['time_num']
            self.ds_net = SurvivalNet(self.time_num, latent_size, dropout_p=ds_drop_p)
            self.survival_loss = config["survival_loss"]
            if self.survival_loss == 'MTLR':
                self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
                self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)   
            self.survival_T_max = config['survival_T_max']
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--ds_drop_p", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--ds_lr", type=float, default=1e-4)
        parser.add_argument("--ds_weight_decay", type=float, default=1e-4)
        parser.add_argument("--ds_beta1", type=float, default=0.5)
        parser.add_argument('--ds_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine]')
        parser.add_argument('--ds_epoch_num_decay', type=int, default=50,
                            help='Number of epoch to linearly decay learning rate to zero (lr_policy == linear)')
        parser.add_argument('--ds_decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch (lr_policy == step)')
        parser.add_argument("--cl_loss", type=str, default="wbce", help="Loss function to use. Options: wbce, bce")
        parser.add_argument("--ds_task", type=str, default='class', 
                            help='downstream task, options: class (cancer type classification), surv (survival analysis)')
        parser.add_argument('--survival_loss', type=str, default='MTLR', help='choose the survival loss')
        parser.add_argument('--survival_T_max', type=float, default=-1, help='maximum T value for survival prediction task')
        parser.add_argument('--time_num', type=int, default=256, help='number of time intervals in the survival model')
        parser.add_argument('--ds_latent_agg_method', type=str, default='mean',
                                help='method to aggregate latent representations from autoencoders of A, B and C, options: "mean", "concat", "sum"')
        parser.add_argument('--ds_freeze_ae', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to freeze the autoencoder for downstream model')
        parser.add_argument('--ds_save_latent_testing', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of testing data')
        parser.add_argument('--ds_mask_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from A will be masked')
        parser.add_argument('--ds_mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from B will be masked')
        parser.add_argument('--ds_masking_method', type=str, default='zero',
                                help='method to mask A, options: "zero", "gaussian_noise"')
        return parent_parser

    def forward(self, x):
        if self.ae_net == 'ae':
            h_A, h_B, h_C = self.feature_extractor(x)
            if self.ds_latent_agg_method == 'concat':
                h = torch.cat([h_A, h_B, h_C], dim=1)
            elif self.ds_latent_agg_method == 'mean':
                h = torch.mean(torch.stack([h_A, h_B, h_C]), axis=0)
            elif self.ds_latent_agg_method == 'sum':
                h = torch.sum(torch.stack([h_A, h_B, h_C]), axis=0)
        elif self.ae_net == 'vae':
            h, _, _, _ = self.feature_extractor(x)
        return h, self.ds_net(h)

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.ds_lr, weight_decay=self.ds_weight_decay, betas=(self.ds_beta1, 0.999))
        if self.ds_lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_lambda = 1.0 - max(0, epoch - self.ds_max_epochs + self.ds_epoch_num_decay) / float(self.ds_epoch_num_decay + 1)
                return lr_lambda
            # lr_scheduler is imported from torch.optim
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.ds_lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.ds_decay_step_size, gamma=0.1)
        elif self.ds_lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif self.ds_lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.ds_max_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def shared_step(self, batch):
        if self.ds_task == 'class':
            output_dict = self.class_step(batch)
        elif self.ds_task == 'surv':
            output_dict = self.surv_step(batch)
        return output_dict

    def class_step(self, batch):
        x_A, x_B, x_C = batch['x']
        y = batch['y']
        sample_ids = batch['sample_id']
        if self.ds_mask_A:
            x_A = self.mask_x(x_A)
        if self.ds_mask_B:
            x_B = self.mask_x(x_B)
        h, y_out = self.forward((x_A, x_B, x_C))
        if self.cl_loss == "wbce":
            down_loss = self.wbce(y_out, y, self.class_weights)
        elif self.cl_loss == "bce":
            down_loss = self.criterion(y_out, y)
        y_true = y.long()
        y_prob = F.softmax(y_out, dim=1)
        _, y_pred = torch.max(y_prob, 1)
        return {
            "loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true": y_true,
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach()
        }

    def mask_x(self, x):
        x_masked = []
        for i in range(len(x)):
            if self.ds_masking_method == 'zero':
                x_masked.append(torch.zeros_like(x[i]))
            elif self.ds_masking_method == 'gaussian_noise':
                x_masked.append(x[i] + torch.randn_like(x[i]))
        return x_masked
    
    def surv_step(self, batch):
        x_A, x_B, x_C = batch['x']
        sample_ids = batch['sample_id']
        surv_T, surv_E, y_true = batch['survival']
        h, y_out = self.forward((x_A, x_B, x_C))
        if self.survival_loss == 'MTLR':
            down_loss = MTLR_survival_loss(y_out, y_true, surv_E, self.tri_matrix_1)
        predict = self.predict_risk(y_out)
        survival = predict['survival']
        risk = predict['risk']
        return {
            "loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true_E": surv_E.detach(),
            "y_true_T": surv_T.detach(),
            "survival": survival.detach(),
            "risk": risk.detach(),
            "y_out": y_out.detach()
        }

    def compute_class_metrics(self, outputs):
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
        return accuracy, precision, recall, f1, auc

    def compute_surv_metrics(self, outputs):
        y_true_E = torch.cat([x["y_true_E"] for x in outputs]).cpu().numpy()
        y_true_T = torch.cat([x["y_true_T"] for x in outputs]).cpu().numpy()
        y_pred_risk = torch.cat([x["risk"] for x in outputs]).cpu().numpy()
        y_pred_survival = torch.cat([x["survival"] for x in outputs]).cpu().numpy()
        time_points = self.get_time_points(self.survival_T_max, self.time_num)
        try:
            c_index = metrics.c_index(y_true_T, y_true_E, y_pred_risk)
        except ValueError:
            c_index = -1
            print('ValueError: NaNs detected in input when calculating c-index.')
        try:
            ibs = metrics.ibs(y_true_T, y_true_E, y_pred_survival, time_points)
        except ValueError:
            ibs = -1
            print('ValueError: NaNs detected in input when calculating integrated brier score.')
        return c_index, ibs
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_down_loss", avg_loss)
        if self.ds_task == 'class':
            accuracy, precision, recall, f1, auc = self.compute_class_metrics(outputs)
            self.log("train_accuracy", accuracy)
            self.log("train_precision", precision)
            self.log("train_recall", recall)
            self.log("train_f1", f1)
            self.log("train_auc", auc)
        
        if self.ds_task == 'surv':
            c_index, ibs = self.compute_surv_metrics(outputs)
            self.log("train_c_index", c_index)
            self.log("train_ibs", ibs)
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('val_accuracy', summary='max')
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_down_loss", avg_down_loss)
        if self.ds_task == 'class':
            accuracy, precision, recall, f1, auc = self.compute_class_metrics(outputs)
            self.log("val_accuracy", accuracy)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)
            self.log("val_auc", auc)
        elif self.ds_task == 'surv':
            c_index, ibs = self.compute_surv_metrics(outputs)
            self.log("val_c_index", c_index)
            self.log("val_ibs", ibs)
    
    def test_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('test_accuracy', summary='max')
        return self.shared_step(batch)
    
    def test_epoch_end(self, outputs):
        avg_down_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_down_loss", avg_down_loss)

        if self.ds_task == 'class':
            accuracy, precision, recall, f1, auc = self.compute_class_metrics(outputs)
            self.log("test_accuracy", accuracy)
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_f1", f1)
            self.log("test_auc", auc)
        elif self.ds_task == 'surv':
            c_index, ibs = self.compute_surv_metrics(outputs)
            self.log("test_c_index", c_index)
            self.log("test_ibs", ibs)
        if self.ds_save_latent_testing:
            sample_ids_list = []
            for x in outputs:
                sample_ids_list.extend(x["sample_ids"])
            h_concat = torch.cat([x["h"] for x in outputs]).cpu().numpy()
            latent_space = pd.DataFrame(h_concat, index=sample_ids_list)
            latent_space.to_csv(os.path.join(self.checkpoint_path, 'latent_space.tsv'), sep='\t')
    
    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def on_predict_epoch_end(self, results):
        outputs = results[0]
        if self.ds_task == 'class':
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
        
        elif self.ds_task == "surv":
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
        
        return preds

    
    def get_tri_matrix(self, dimension_type=1):
        """
        Get tensor of the triangular matrix
        """
        if dimension_type == 1:
            ones_matrix = torch.ones(self.time_num, self.time_num + 1, device=self.device)
        else:
            ones_matrix = torch.ones(self.time_num + 1, self.time_num + 1, device=self.device)
        tri_matrix = torch.tril(ones_matrix).cuda()
        return tri_matrix
    
    def predict_risk(self, y_out):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        if self.survival_loss == 'MTLR':
            phi = torch.exp(torch.mm(y_out, self.tri_matrix_1))
            div = torch.repeat_interleave(torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1)

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {'density': density, 'survival': survival, 'hazard': hazard, 'risk': risk}
    
    def get_time_points(self, T_max, extra_time_percent=0.1):
        """
        Get time points for the MTLR model
        """
        # Get time points in the time axis
        time_points = np.linspace(0, T_max * (1 + extra_time_percent), self.time_num + 1)

        return time_points
    