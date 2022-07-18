from email.policy import default
from pickletools import optimize
from random import sample
from sched import scheduler
from turtle import forward
from pl_bolts import optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .networks import SimCLRProjectionHead, CLIPProjectionHead, AESepB, AESepA, AESepAB, ClassifierNet, VAESepB, VAESepAB, SurvivalNet, RegressionNet
from .losses import SimCLR_Loss, weighted_binary_cross_entropy, CLIPLoss, BarlowTwinsLoss, MTLR_survival_loss, NTXentLoss, SimSiamLoss, mmd_rbf
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
import string

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, ae_net, ae_weight_kl, latent_size, projection_size, ae_lr, ae_weight_decay, ae_momentum, ae_drop_p, ae_beta1, ae_lr_policy, ae_epoch_num_decay, ae_decay_step_size, max_epochs, cont_align_loss_criterion, cont_loss_temp, cont_loss_lambda, ae_optimizer, ae_use_lrscheduler, cont_align_loss_weight, split_A, split_B, mask_A, mask_B, num_mask_A, num_mask_B, masking_method, batch_size, ae_dim_1B, ae_dim_2B, ae_dim_1A, ae_dim_2A, ae_dim_1C, ae_dim_2C, **config):
        super(AutoEncoder, self).__init__()
        self.input_size_A = input_size_A
        self.input_size_B = input_size_B
        self.input_size_C = input_size_C
        self.batch_size = batch_size
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
        self.recon_loss_criterion = config['recon_loss_criterion']
        self.cont_align_loss_weight = cont_align_loss_weight
        self.cont_noise_loss_weight = config['cont_noise_loss_weight']
        self.cont_loss_temp = cont_loss_temp
        self.cont_loss_lambda = cont_loss_lambda
        self.add_distance_loss_to_latent = config['add_distance_loss_to_latent']
        self.add_distance_loss_to_proj = config['add_distance_loss_to_proj']
        self.distance_loss_weight = config['distance_loss_weight']
        self.add_consistency_loss = config['add_consistency_loss']
        self.consistency_loss_weight = config['consistency_loss_weight']
        self.ae_optimizer = ae_optimizer
        self.ae_use_lrscheduler = ae_use_lrscheduler
        self.cont_align_loss_criterion = cont_align_loss_criterion
        self.cont_noise_loss_criterion = config['cont_noise_loss_criterion']
        self.cont_align_loss_latent = config['cont_align_loss_latent']
        self.cont_loss_similarity = config['cont_loss_similarity']
        self.cont_loss_normalize = config['cont_loss_normalize']
        self.cont_loss_p_norm = config['cont_loss_p_norm']
        self.add_cont_type_loss = config['add_cont_type_loss']
        self.split_A = split_A
        self.mask_A = mask_A
        self.num_mask_A = num_mask_A
        self.split_B = split_B
        self.mask_B = mask_B
        self.num_mask_B = num_mask_B
        self.mask_C = config['mask_C']
        self.ratio_mask_C = config['ratio_mask_C']
        self.masking_method = masking_method
        self.choose_masking_method_every_epoch = config['choose_masking_method_every_epoch']
        self.use_one_encoder = config['use_one_encoder']
        self.use_one_decoder = config['use_one_decoder']
        self.concat_latent_for_decoder = config['concat_latent_for_decoder']
        self.recon_all_thrice = config['recon_all_thrice']
        self.predict_masked_chromosomes = config['predict_masked_chromosomes']
        self.use_rep_trick = config['use_rep_trick']
        if self.ae_net == "ae":
            if self.split_A and self.split_B:
                self.ae_dim_1B = 128
                self.ae_dim_1A = 128
                self.ae = AESepAB((input_size_A, input_size_B, input_size_C), latent_size, self.use_one_encoder, self.use_one_decoder, self.concat_latent_for_decoder, self.recon_all_thrice, self.use_rep_trick, dropout_p=ae_drop_p, dim_1B=self.ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=self.ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
            elif self.split_A:
                self.ae_dim_1A = 128
                self.ae_dim_1B = 1024
                self.ae = AESepA((input_size_A, input_size_B, input_size_C), latent_size, self.use_one_decoder, self.concat_latent_for_decoder, self.recon_all_thrice, dropout_p=ae_drop_p, dim_1B=self.ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=self.ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
            elif self.split_B:
                self.ae_dim_1B = 128
                self.ae_dim_1A = 1024
                self.ae = AESepB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=self.ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=self.ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        elif self.ae_net == "vae":
            if self.split_A and self.split_B:
                self.ae = VAESepAB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
            elif self.split_B:
                self.ae = VAESepB((input_size_A, input_size_B, input_size_C), latent_size, dropout_p=ae_drop_p, dim_1B=ae_dim_1B, dim_2B=ae_dim_2B, dim_1A=ae_dim_1A, dim_2A=ae_dim_2A, dim_1C=ae_dim_1C, dim_2C=ae_dim_2C)
        
        if self.recon_loss_criterion != "none":
            if self.recon_loss_criterion == "mse":
                self.recon_criterion = nn.MSELoss()
            elif self.recon_loss_criterion == "l1":
                self.recon_criterion = nn.L1Loss()
            elif self.recon_loss_criterion == 'bce':
                self.recon_criterion = nn.BCELoss()

        self.projection_size = latent_size // 2
        if self.cont_align_loss_criterion != "none":
            if self.cont_align_loss_criterion == "simclr":
                # self.cont_align_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
                self.projector = SimCLRProjectionHead(latent_size, latent_size, self.projection_size)
                self.cont_align_criterion = losses.NTXentLoss(temperature=cont_loss_temp)
            elif self.cont_align_loss_criterion == "clip":
                self.projector = CLIPProjectionHead(latent_size, self.projection_size, ae_drop_p)
                self.cont_align_criterion = CLIPLoss(temperature = cont_loss_temp, latent_size = latent_size, proj_size = self.projection_size)
            elif self.cont_align_loss_criterion == "barlowtwins":
                self.cont_align_criterion = BarlowTwinsLoss(lambd=cont_loss_lambda, latent_size=latent_size, proj_size=self.projection_size)
            elif self.cont_align_loss_criterion == 'ntxent':
                self.cont_align_criterion = NTXentLoss(latent_dim = latent_size, temperature=cont_loss_temp, batch_size=self.batch_size, similarity=self.cont_loss_similarity, normalize=self.cont_loss_normalize, p_norm=self.cont_loss_p_norm)
            elif self.cont_align_loss_criterion == 'simsiam':
                self.cont_align_criterion = SimSiamLoss(latent_dim=latent_size)
        
        if self.cont_noise_loss_criterion != "none":
            if self.cont_noise_loss_criterion == "simclr":
                # self.cont_noise_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
                self.projector = SimCLRProjectionHead(latent_size, latent_size, self.projection_size)
                self.cont_noise_criterion = losses.NTXentLoss(temperature=cont_loss_temp)
            elif self.cont_noise_loss_criterion == "clip":
                self.projector = CLIPProjectionHead(latent_size, self.projection_size, ae_drop_p)
                self.cont_noise_criterion = CLIPLoss(temperature = cont_loss_temp, latent_size = latent_size, proj_size = self.projection_size)
            elif self.cont_noise_loss_criterion == "barlowtwins":
                self.cont_noise_criterion = BarlowTwinsLoss(lambd=cont_loss_lambda, latent_size=latent_size, proj_size=self.projection_size)
            elif self.cont_noise_loss_criterion == 'ntxent':
                self.cont_noise_criterion = NTXentLoss(latent_dim = latent_size, temperature=cont_loss_temp, batch_size=self.batch_size, similarity=self.cont_loss_similarity, normalize=self.cont_loss_normalize, p_norm=self.cont_loss_p_norm)
            elif self.cont_noise_loss_criterion == 'simsiam':
                self.cont_noise_criterion = SimSiamLoss(latent_dim=latent_size)
        
        if self.add_distance_loss_to_latent or self.add_distance_loss_to_proj:
            self.distance_loss_criterion = config['distance_loss_criterion']
            if self.distance_loss_criterion == 'mse':
                self.dist_loss = nn.MSELoss()
            elif self.distance_loss_criterion == 'l1':
                self.dist_loss = nn.L1Loss()
            elif self.distance_loss_criterion == 'bce':
                self.dist_loss = nn.BCELoss()
        
        if self.add_consistency_loss:
            self.cons_loss = nn.MSELoss()
        
        if self.mask_B:
            self.mask_B_ids = np.random.randint(0, len(self.input_size_B), size=self.num_mask_B)
        if self.mask_A:
            self.mask_A_ids = np.random.randint(0, len(self.input_size_A), size=self.num_mask_A)
        if self.mask_C:
            self.mask_C_features = np.random.randint(0, self.input_size_C, size=int(self.ratio_mask_C * self.input_size_C))
        
        self.change_ch_to_mask_every_epoch = config['change_ch_to_mask_every_epoch']

        if self.predict_masked_chromosomes:
            num_mask_total = 1
            if self.split_A:
                num_mask_total += len(self.input_size_A)
            else:
                num_mask_total += 1
            if self.split_B:
                num_mask_total += len(self.input_size_B)
            else:
                num_mask_total += 1
            self.masked_chr_prediction_weight = config['masked_chr_prediction_weight']
            self.mask_pred_net = ClassifierNet(num_mask_total, latent_size * 3, dropout_p=ae_drop_p)
            self.mask_pred_criterion = nn.CrossEntropyLoss()
        
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
        parser.add_argument("--recon_loss_criterion", type=str, default="mse",
                            help="Reconstruction loss criterion, options: [none, mse, l1, bce]") 
        parser.add_argument("--cont_align_loss_criterion", type=str, default="barlowtwins", help="contrastive alignment loss to use, options: none, simclr, clip, barlowtwins, ntxent, simsiam")
        parser.add_argument("--cont_noise_loss_criterion", type=str, default="barlowtwins", help="contrastive noise loss to use, options: none, clip, barlowtwins, ntxent")
        parser.add_argument("--add_cont_type_loss", default=False, type=lambda x: (str(x).lower() == 'true')
                            , help="Add cancer type wise contrastive loss")
        parser.add_argument("--cont_loss_similarity", type=str, default="cosine", help="similarity function to use for ntxent loss, options: [cosine, dot]")
        parser.add_argument("--cont_loss_normalize", default=False, type=lambda x: (str(x).lower() == 'true'), help="whether to normalize ntxent loss")
        parser.add_argument("--cont_loss_p_norm", type=float, default=2.0, help="p-norm to use for ntxent loss if cont_loss_normalize is set to true")
        parser.add_argument("--cont_loss_temp", type=float, default=0.1)
        parser.add_argument("--cont_loss_lambda", type=float, default=0.0051, help="for barlowtwins")
        parser.add_argument("--cont_align_loss_weight", type=float, default=0.5)
        parser.add_argument("--cont_noise_loss_weight", type=float, default=0.5)
        parser.add_argument("--cont_align_loss_latent", type=str, default="masked", help="the latent representation on which contrastive alignment loss should be computed, options=['masked', 'unmasked', 'mean']")
        parser.add_argument("--add_distance_loss_to_latent", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--add_distance_loss_to_proj", default=False, type=lambda x: (str(x).lower() == 'true'), help="works only when a constrastive loss is used")
        parser.add_argument("--distance_loss_weight", type=float, default=0.5)
        parser.add_argument("--distance_loss_criterion", type=str, default="mse", help="distance loss to use, options: mse, bce, l1")
        parser.add_argument("--add_consistency_loss", default=False, type=lambda x: (str(x).lower() == 'true'), help="add consistency loss")
        parser.add_argument("--consistency_loss_weight", type=float, default=10.0)
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
        parser.add_argument('--mask_C', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, ratio_mask_C of C are masked')
        parser.add_argument('--ratio_mask_C', type=float, default=0.0,
                                help='ratio of C to mask')
        parser.add_argument('--masking_method', type=str, default='zero',
                                help='method to mask data, can be "zero", "gaussian_noise", or "swap_noise"')
        parser.add_argument('--choose_masking_method_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the masking method is chosen randomly each epoch and "masking_method" argument is ignored')
        parser.add_argument('--change_ch_to_mask_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the chromosomes to mask are changed each epoch')
        parser.add_argument('--predict_masked_chromosomes', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the chromosomes that are masked will be predicted')
        parser.add_argument('--masked_chr_prediction_weight', type=float, default=0.5,
                                help='weight of masked chromosomes prediction loss')
        parser.add_argument('--recon_all_thrice', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, modalities A, B and C will be reconstructed from latent representations of each A, B and C modalities')
        parser.add_argument('--use_one_encoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, only one encoder is used to represent all modalities')
        parser.add_argument('--use_one_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, only one decoder is used to reconstruct all modalities')
        parser.add_argument('--concat_latent_for_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, latent vectors from A, B and C are concatenated before being fed into the decoder')
        parser.add_argument('--use_rep_trick', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='use reparameterization in trick in ae')
        parser.add_argument('--pretraining_max_epochs', type=int, default=50, help='maximum number of epochs for pretraining')
        parser.add_argument('--pretraining_patience', type=int, default=35, help='patience for pretraining')
        parser.add_argument('--add_MMD_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add MMD loss')
        parser.add_argument('--MMD_loss_weight', type=float, default=0.5,
                                help='weight of MMD loss')
        parser.add_argument('--add_latent_reconstruction_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add latent reconstruction loss')
        parser.add_argument('--latent_reconstruction_loss_weight', type=float, default=0.5, 
                                help='weight of latent reconstruction loss')
        return parent_parser

    def forward(self, x):
        if self.ae_net == "vae":
            return self.ae(x)
        elif self.ae_net == "ae":
            return self.ae.encode(x)

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
            if self.mask_C:
                self.mask_C_features = np.random.randint(0, self.input_size_C, size=int(self.ratio_mask_C * self.input_size_C))
        if self.choose_masking_method_every_epoch:
            self.masking_method = np.random.choice(['zero', 'gaussian_noise', 'swap_noise'])

    def mask_x_ch(self, x, mask_ids):
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
    
    def mask_x_feat(self, x, mask_feats):
        if self.masking_method == 'zero':
            x[:, mask_feats] = torch.zeros_like(x[:, mask_feats])
        elif self.masking_method == 'gaussian_noise':
            x[:, mask_feats] = x[:, mask_feats] + torch.randn_like(x[:, mask_feats])
        elif self.masking_method == 'swap_noise':
            for j in range(x.shape[1]):
                x[:, j] = x[torch.randperm(x.shape[0]), j]
        return x
    
    def sum_subset_losses(self, x_recon, x):
        x_recon_loss = []
        for i in range(len(x)):
            x_recon_loss.append(self.recon_criterion(x_recon[i], x[i]))
        return sum(x_recon_loss)
    
    def sum_losses(self, x_A_recon, x_B_recon, x_C_recon, x_A, x_B, x_C):
        if self.split_A:
            recon_A_loss = self.sum_subset_losses(x_A_recon, x_A)
        else:
            recon_A_loss = self.recon_criterion(x_A_recon, x_A)
        if self.split_B:
            recon_B_loss = self.sum_subset_losses(x_B_recon, x_B)
        else:
            recon_B_loss = self.recon_criterion(x_B_recon, x_B)
        recon_C_loss = self.recon_criterion(x_C_recon, x_C)
        if self.mask_A and self.mask_B:
            recon_loss = 0.4 * recon_A_loss + 0.4 * recon_B_loss + 0.2 * recon_C_loss
        elif self.mask_A:
            recon_loss = 0.5 * recon_A_loss + 0.25 * recon_B_loss + 0.25 * recon_C_loss 
        elif self.mask_B:
            recon_loss = 0.5 * recon_B_loss + 0.25 * recon_A_loss + 0.25 * recon_C_loss
        else:
            recon_loss = recon_A_loss + recon_B_loss + recon_C_loss
        return recon_loss

    def ae_step(self, batch):
        logs = {}
        pretext_loss = 0
        x_A, x_B, x_C = batch['x']
        x_A_in, x_B_in, x_C_in = x_A, x_B, x_C
        if self.mask_A:
            x_A_in = self.mask_x_ch(x_A, self.mask_A_ids)
        if self.mask_B:
            x_B_in = self.mask_x_ch(x_B, self.mask_B_ids)
        if self.mask_C:
            x_C_in = self.mask_x_feat(x_C, self.mask_C_features)
        h_A, h_B, h_C = self.ae.encode((x_A_in, x_B_in, x_C_in))
        h_A_unmasked, h_B_unmasked, h_C_unmasked = self.ae.encode((x_A, x_B, x_C))
        if self.cont_noise_loss_criterion != "none":
            cont_noise_loss = 0
            for omics_type, h_masked, h_unmasked in zip(['A', 'B', 'C'], [h_A, h_B, h_C], [h_A_unmasked, h_B_unmasked, h_C_unmasked]):
                cont_noise_logs_type, cont_noise_loss_type = self.cont_noise_step(h_masked, h_unmasked, omics_type)
                logs.update(cont_noise_logs_type)
                cont_noise_loss += cont_noise_loss_type
            logs['{}_cont_noise_loss_total'.format(self.mode)] = cont_noise_loss
            pretext_loss += (cont_noise_loss / 3) * self.cont_noise_loss_weight
        if self.predict_masked_chromosomes:
            h = torch.cat((h_A, h_B, h_C), dim=1)
            mask_y_out = self.mask_pred_net(h)
            mask_y = torch.zeros(mask_y_out.shape).to(self.device)
            if self.mask_A:
                mask_y[:, self.mask_A_ids] = 1
            if self.mask_B:
                mask_y[:, self.mask_B_ids] = 1
            mask_pred_loss = self.mask_pred_criterion(mask_y_out, mask_y)
            pretext_loss += self.masked_chr_prediction_weight * mask_pred_loss
            logs['{}_mask_pred_loss'.format(self.mode)] = mask_pred_loss
        if self.recon_loss_criterion != "none":
            if self.recon_all_thrice:
                recon_list = self.ae.decode((h_A, h_B, h_C))
            else:
                recon_list.append(self.ae.decode((h_A, h_B, h_C)))
            recon_loss = 0
            for i, x_recon in enumerate(recon_list):
                x_A_recon, x_B_recon, x_C_recon = x_recon
                recon_loss_all = self.sum_losses(x_A_recon, x_B_recon, x_C_recon, x_A, x_B, x_C)
                recon_loss += recon_loss_all
                if self.recon_all_thrice:
                    logs['{}_recon_all_from_{}_loss'.format(self.mode, string.ascii_uppercase[i])] = recon_loss_all
                else:
                    logs['{}_recon_all_loss'.format(self.mode)] = recon_loss_all
            pretext_loss += recon_loss
            if self.recon_all_thrice:
                logs['{}_total_recon_all_loss'.format(self.mode)] = recon_loss
        
        if self.cont_align_loss_latent == 'unmasked':
            h_A, h_B, h_C = h_A_unmasked, h_B_unmasked, h_C_unmasked
        elif self.cont_align_loss_latent == 'mean':
            h_A, h_B, h_C = (h_A + h_A_unmasked) / 2, (h_B + h_B_unmasked) / 2, (h_C + h_C_unmasked) / 2
        
        return logs, (h_A, h_B, h_C), pretext_loss 
    
    def vae_step(self, batch):
        logs = {}
        x_A, x_B, x_C = batch['x']
        if self.mask_B:
            x_B_masked = self.mask_x_ch(x_B, self.mask_B_ids)
            z, recon_x, mean, log_var = self.ae((x_A, x_B_masked, x_C))
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = self.recon_criterion(recon_x[0], x_A) + recon_B_loss + self.recon_criterion(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_B_loss + kl_loss
            logs['{}_recon_B_loss'.format(self.mode)] = recon_B_loss
            logs['{}_recon_B_kl_loss'.format(self.mode)] = recon_loss
        
        elif self.mask_A:
            x_A_masked = self.mask_x_ch(x_A, self.mask_A_ids)
            h, recon_x, mean, log_var = self.ae((x_A_masked, x_B, x_C))
            recon_A_loss = self.sum_subset_losses(recon_x[0], x_A)
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = recon_A_loss + recon_B_loss + self.recon_criterion(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_A_loss + kl_loss
            logs['{}_recon_A_loss'.format(self.mode)] = recon_A_loss
            logs['{}_recon_A_kl_loss'.format(self.mode)] = recon_loss

        else:
            h, recon_x, mean, log_var = self.ae((x_A, x_B, x_C))
            recon_A_loss = self.recon_criterion(recon_x[0], x_A)
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_C_loss = self.recon_criterion(recon_x[2], x_C)
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
    
    def cont_align_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        if self.cont_align_loss_criterion == "clip":
            loss_A_B, loss_A1, loss_B1 = self.cont_align_criterion(h_A, h_B)
            loss_B_C, loss_B2, loss_C1 = self.cont_align_criterion(h_B, h_C)
            loss_C_A, loss_C2, loss_A2 = self.cont_align_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_A = (loss_A1 + loss_A2) / 2
            loss_B = (loss_B1 + loss_B2) / 2
            loss_C = (loss_C1 + loss_C2) / 2
            logs['{}_cont_{}_loss_A'.format(self.mode, self.cont_pair)] = loss_A
            logs['{}_cont_{}_loss_B'.format(self.mode, self.cont_pair)] = loss_B
            logs['{}_cont_{}_loss_C'.format(self.mode, self.cont_pair)] = loss_C

        elif self.cont_align_loss_criterion == "barlowtwins":
            loss_A_B, loss_on_diag1, loss_off_diag1, z_A, z_B = self.cont_align_criterion(h_A, h_B)
            loss_B_C, loss_on_diag2, loss_off_diag2, _, z_C = self.cont_align_criterion(h_B, h_C)
            loss_C_A, loss_on_diag3, loss_off_diag3, _, _ = self.cont_align_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_on_diag = (loss_on_diag1 + loss_on_diag2 + loss_on_diag3) / 3
            loss_off_diag = (loss_off_diag1 + loss_off_diag2 + loss_off_diag3) / 3
            logs['{}_cont_{}_loss_on_diag'.format(self.mode, self.cont_pair)] = loss_on_diag
            logs['{}_cont_{}_loss_off_diag'.format(self.mode, self.cont_pair)] = loss_off_diag
        
        elif self.cont_align_loss_criterion == "simclr":
            z_A = self.projector(h_A)
            z_B = self.projector(h_B)
            z_C = self.projector(h_C)
            z_AB = torch.cat((z_A, z_B), dim=0)
            z_BC = torch.cat((z_B, z_C), dim=0)
            z_CA = torch.cat((z_C, z_A), dim=0)
            labels = torch.arange(z_A.shape[0]).repeat(2)
            loss_A_B, loss_num1, loss_den1 = self.cont_align_criterion(z_AB, labels)
            loss_B_C, loss_num2, loss_den2 = self.cont_align_criterion(z_BC, labels)
            loss_C_A, loss_num3, loss_den3 = self.cont_align_criterion(z_CA, labels)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_num = (loss_num1 + loss_num2 + loss_num3) / 3
            loss_den = (loss_den1 + loss_den2 + loss_den3) / 3
            logs['{}_cont_{}_loss_num'.format(self.mode, self.cont_pair)] = loss_num
            logs['{}_cont_{}_loss_den'.format(self.mode, self.cont_pair)] = loss_den
        
        elif self.cont_align_loss_criterion == "ntxent":
            loss_A_B, z_A, z_B = self.cont_align_criterion(h_A, h_B)
            loss_B_C, _, z_C = self.cont_align_criterion(h_B, h_C)
            loss_C_A, _, _ = self.cont_align_criterion(h_C, h_A)
            cont_loss = loss_A_B + loss_B_C + loss_C_A
        
        elif self.cont_align_loss_criterion == "simsiam":
            cont_loss, z_A, z_B, z_C = self.cont_align_criterion(h_A, h_B, h_C)

        logs['{}_cont_{}_loss'.format(self.mode, self.cont_pair)] = cont_loss
        if self.add_distance_loss_to_proj:
            _, dist_loss = self.dist_step((z_A, z_B, z_C))
            cont_loss += dist_loss * self.distance_loss_weight
            self.log('{}_dist_loss_btw_proj'.format(self.mode), dist_loss, on_step=False, on_epoch=True)
        return logs, cont_loss
    
    def MMD_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        MMD_loss = 0
        MMD_loss += mmd_rbf(h_A, h_B)
        MMD_loss += mmd_rbf(h_B, h_C)
        MMD_loss += mmd_rbf(h_C, h_A)
        return logs, MMD_loss
    
    def latent_recon_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        latent_recon_loss = 0
        if self.hparams.recon_all_thrice:
            recon_list = self.ae.decode((h_A, h_B, h_C))
            x_recon = (recon_list[0][0], recon_list[1][1], recon_list[2][2])
            h_A_recon, h_B_recon, h_C_recon = self.ae.encode(x_recon)
        latent_recon_loss += self.recon_criterion(h_A, h_A_recon)
        latent_recon_loss += self.recon_criterion(h_B, h_B_recon)
        latent_recon_loss += self.recon_criterion(h_C, h_C_recon)
        return logs, latent_recon_loss
    
    def cont_noise_step(self, h_masked, h_unmasked, omics_type='A'):
        logs = {}
        if self.cont_noise_loss_criterion == "clip":
            cont_noise_loss, loss_masked, loss_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
            # logs['{}_cont_noise_loss_{}_masked'.format(self.mode, omics_type)] = loss_masked
            # logs['{}_cont_noise_loss_{}_unmasked'.format(self.mode, omics_type)] = loss_unmasked
        elif self.cont_noise_loss_criterion == "barlowtwins":
            cont_noise_loss, loss_on_diag, loss_off_diag, z_masked, z_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
            # logs['{}_cont_noise_{}_loss_on_diag'.format(self.mode, omics_type)] = loss_on_diag
            # logs['{}_cont_noise_{}_loss_off_diag'.format(self.mode, omics_type)] = loss_off_diag
        elif self.cont_noise_loss_criterion == "ntxent":
            cont_noise_loss, z_masked, z_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
        logs['{}_cont_noise_loss_{}'.format(self.mode, omics_type)] = cont_noise_loss
        return logs, cont_noise_loss
        
    def dist_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        if self.distance_loss_criterion == 'bce':
            h_A = torch.clamp(h_A, min=1e-7, max=0.9999)
            h_B = torch.clamp(h_B, min=1e-7, max=0.9999)
            h_C = torch.clamp(h_C, min=1e-7, max=0.9999)
        dist_loss = self.dist_loss(h_A, h_B) + self.dist_loss(h_B, h_C) + self.dist_loss(h_C, h_A)
        return logs, dist_loss
    
    def cons_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        cons_loss = 0
        h_A_recon_using_dec_B = self.ae.encode_B(self.ae.decode_h_B(h_A)[1])
        h_B_recon_using_dec_A = self.ae.encode_A(self.ae.decode_h_A(h_B)[0])
        h_C_recon_using_dec_A = self.ae.encode_A(self.ae.decode_h_A(h_C)[0])
        h_A_recon_using_dec_C = self.ae.encode_C(self.ae.decode_h_C(h_A)[2])
        h_B_recon_using_dec_C = self.ae.encode_C(self.ae.decode_h_C(h_B)[2])
        h_C_recon_using_dec_B = self.ae.encode_B(self.ae.decode_h_B(h_C)[1])
        cons_loss += self.cons_loss(h_A_recon_using_dec_B, h_B)
        cons_loss += self.cons_loss(h_B_recon_using_dec_A, h_A)
        cons_loss += self.cons_loss(h_C_recon_using_dec_A, h_A)
        cons_loss += self.cons_loss(h_A_recon_using_dec_C, h_C)
        cons_loss += self.cons_loss(h_B_recon_using_dec_C, h_C)
        cons_loss += self.cons_loss(h_C_recon_using_dec_B, h_B)
        return logs, cons_loss
    
    def training_step(self, batch, batch_idx):
        if self.ae_net == 'ae':
            logs, h, pretext_loss = self.ae_step(batch) 
        elif self.ae_net == 'vae':
            logs, h, pretext_loss = self.vae_step(batch)
        for k, v in logs.items():
            self.log(k, v, on_step=False, on_epoch=True)
        if self.add_distance_loss_to_latent:
            logs, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.distance_loss_weight
            self.log('{}_dist_loss_btw_latent'.format(self.mode), dist_loss, on_step=False, on_epoch=True)
        if self.add_consistency_loss:
            logs, cons_loss = self.cons_step(h)
            pretext_loss += cons_loss * self.consistency_loss_weight
            self.log('{}_cons_loss'.format(self.mode), cons_loss, on_step=False, on_epoch=True)
        if self.hparams.add_MMD_loss:
            logs, MMD_loss = self.MMD_step(h)
            pretext_loss += MMD_loss * self.hparams.MMD_loss_weight
            self.log('{}_MMD_loss'.format(self.mode), MMD_loss, on_step=False, on_epoch=True)
        if self.hparams.add_latent_reconstruction_loss:
            logs, latent_recon_loss = self.latent_recon_step(h)
            pretext_loss += latent_recon_loss * self.hparams.latent_reconstruction_loss_weight
            self.log('{}_latent_recon_loss'.format(self.mode), latent_recon_loss, on_step=False, on_epoch=True)
        if self.cont_align_loss_criterion != "none":
            if self.cont_align_loss_criterion in ['barlowtwins', 'clip'] or h[0].shape[0] == self.batch_size:
                self.cont_pair = 'align'
                logs, cont_loss = self.cont_align_step(h)
                for k, v in logs.items():
                    self.log(k, v, on_step=False, on_epoch=True)
                pretext_loss += self.cont_align_loss_weight * cont_loss
                if self.add_cont_type_loss:
                    h_list = self.prepare_cont_type_h(h, batch['y'])
                    for i, h_type in enumerate(h_list):
                        self.cont_pair = 'align_type'
                        logs, cont_loss = self.cont_align_step(h_type)
                        for k, v in logs.items():
                            self.log(k, v, on_step=False, on_epoch=True)
                        pretext_loss += self.cont_align_loss_weight * cont_loss
        self.log('{}_pretext_loss'.format(self.mode), pretext_loss, on_step=False, on_epoch=True)
        return pretext_loss
    
    def validation_step(self, batch, batch_idx):
        # if self.global_step == 0: 
        #     wandb.define_metric('val_pretext_loss', summary='min')
        #     wandb.define_metric('val_recon_loss', summary='min')
        self.mode = 'val'
        pretext_loss = 0
        if self.ae_net == 'ae':
            logs, h, recon_loss = self.ae_step(batch) 
        elif self.ae_net == 'vae':
            logs, h, recon_loss = self.vae_step(batch)
        pretext_loss += recon_loss
        if self.add_distance_loss_to_latent:
            _, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.distance_loss_weight
            logs['{}_dist_loss'.format(self.mode)] = dist_loss
        if self.add_consistency_loss:
            _, cons_loss = self.cons_step(h)
            pretext_loss += cons_loss * self.consistency_loss_weight
            logs['{}_cons_loss'.format(self.mode)] = cons_loss
        if self.hparams.add_MMD_loss:
            _, MMD_loss = self.MMD_step(h)
            pretext_loss += MMD_loss * self.hparams.MMD_loss_weight
            logs['{}_MMD_loss'.format(self.mode)] = MMD_loss
        if self.hparams.add_latent_reconstruction_loss:
            _, latent_recon_loss = self.latent_recon_step(h)
            pretext_loss += latent_recon_loss * self.hparams.latent_reconstruction_loss_weight
            logs['{}_latent_recon_loss'.format(self.mode)] = latent_recon_loss
        if self.cont_align_loss_criterion != "none":
            cont_logs = {}
            if self.cont_align_loss_criterion in ['barlowtwins', 'clip'] or h[0].shape[0] == self.batch_size:
                self.cont_pair = 'align'
                cont_pair_logs, cont_loss = self.cont_align_step(h)
                cont_logs.update(cont_pair_logs)
                pretext_loss += self.cont_align_loss_weight * cont_loss
                if self.add_cont_type_loss:
                    h_list = self.prepare_cont_type_h(h, batch['y'])
                    for i, h_type in enumerate(h_list):
                        self.cont_pair = 'align_type'
                        cont_pair_logs, cont_loss = self.cont_align_step(h_type)
                        cont_logs.update(cont_pair_logs)
                        pretext_loss += self.cont_align_loss_weight * cont_loss
            logs.update(cont_logs)
        logs['{}_pretext_loss'.format(self.mode)] = pretext_loss
        return logs

    def validation_epoch_end(self, outputs):
        for key, value in outputs[0].items():
            avg = torch.stack([x[key] for x in outputs if key in x.keys()]).mean()
            self.log(key, avg)
        


class DownstreamModel(pl.LightningModule):
    def __init__(self, ae_model_path, A_shape, B_shape, C_shape, class_weights, **config):
        super(DownstreamModel, self).__init__()
        self.ae_model_path = ae_model_path
        self.ds_input_size = config['latent_size']
        self.ds_drop_p = config['ds_drop_p']
        self.num_classes = config['num_classes']
        self.ds_lr = config['ds_lr']
        self.ds_weight_decay = config['ds_weight_decay']
        self.ds_beta1 = config['ds_beta1']
        self.ds_lr_policy = config['ds_lr_policy']
        self.ds_epoch_num_decay = config['ds_epoch_num_decay']
        self.ds_decay_step_size = config['ds_decay_step_size']
        self.class_weights = class_weights
        self.ae_net = config['ae_net']
        self.ds_max_epochs = config['downstream_max_epochs']
        self.ds_task = config['ds_task']
        if self.ds_task == 'multi':
            self.ds_tasks = ['class', 'surv', 'reg']
        else:
            self.ds_tasks = [self.ds_task]
        self.ds_k_class = config['ds_k_class']
        self.ds_k_surv = config['ds_k_surv']
        self.ds_k_reg = config['ds_k_reg']
        self.ds_mask_A = config['ds_mask_A']
        self.ds_mask_B = config['ds_mask_B']
        self.ds_masking_method = config['ds_masking_method']
        if self.ae_model_path is None:
            self.ae_model = AutoEncoder(A_shape, B_shape, C_shape, **config)
        else:
            self.ae_model = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        if config['ds_freeze_ae']:
            self.ae_model.freeze()
        self.ds_latent_agg_method = config['ds_latent_agg_method']
        self.ds_add_omics_identity = config['ds_add_omics_identity']
        if self.ds_add_omics_identity:
            self.ds_latent_agg_method = 'all'
        self.num_classes = config['num_classes']
        self.ds_drop_p = config['ds_drop_p']
        self.cl_loss = config['cl_loss']
        if 'class' in self.ds_tasks:
            self.class_net = ClassifierNet(self.num_classes, self.ds_input_size, dropout_p=self.ds_drop_p)
            self.wbce = weighted_binary_cross_entropy
            self.criterion = nn.CrossEntropyLoss()
            self.cl_loss = self.cl_loss
        if 'surv' in self.ds_tasks:
            self.time_num = config['time_num']
            self.surv_net = SurvivalNet(self.time_num, self.ds_input_size, dropout_p=self.ds_drop_p)
            self.survival_loss = config["survival_loss"]
            if self.survival_loss == 'MTLR':
                self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
                self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)   
            self.survival_T_max = config['survival_T_max']
        if 'reg' in self.ds_tasks:
            self.reg_net = RegressionNet(self.ds_input_size, dropout_p=self.ds_drop_p)
            self.reg_loss = nn.MSELoss()

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
                            help='downstream task, options: class (classification like cancer type classification), surv (survival analysis), reg (regression like age prediction), multi (multi-task training of all 3 tasks together)')
        parser.add_argument("--ds_k_class", type=float, default=1.0, help="Weight for classification loss in multi-task training")
        parser.add_argument("--ds_k_surv", type=float, default=1.0, help="Weight for survival loss in multi-task training")
        parser.add_argument("--ds_k_reg", type=float, default=1.0, help="Weight for regression loss in multi-task training")
        parser.add_argument('--survival_loss', type=str, default='MTLR', help='choose the survival loss')
        parser.add_argument('--survival_T_max', type=float, default=-1, help='maximum T value for survival prediction task')
        parser.add_argument('--time_num', type=int, default=256, help='number of time intervals in the survival model')
        parser.add_argument('--ds_latent_agg_method', type=str, default='mean',
                                help='method to aggregate latent representations from autoencoders of A, B and C, options: "mean", "concat", "sum", "all" (pass all latents one by one)')
        parser.add_argument('--ds_save_latent_pred', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of the prediction dataset')
        parser.add_argument('--ds_save_model_outputs', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the outputs of the downstream model')
        parser.add_argument('--ds_mask_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from A will be masked')
        parser.add_argument('--ds_mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from B will be masked')
        parser.add_argument('--ds_masking_method', type=str, default='zero',
                                help='method to mask A, options: "zero", "gaussian_noise"')
        parser.add_argument('--ds_class_callback_key', type=str, default='accuracy',
                                help='key for the callback to use for classification task')
        parser.add_argument('--ds_surv_callback_key', type=str, default='c_index',
                                help='key for the callback to use for survival task')
        parser.add_argument('--ds_reg_callback_key', type=str, default='mse',
                                help='key for the callback to use for regression task')
        parser.add_argument('--ds_add_omics_identity', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add omics id to latent representations before using them for classification task')
        parser.add_argument('--ds_freeze_ae', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, autoencoder will be frozen')
        parser.add_argument("--load_ds_model", default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='load pretrained downstream model')
        parser.add_argument("--ds_model_path", type=str, default="")
        parser.add_argument('--downstream_max_epochs', type=int, default=150, help='maximum number of epochs for downstream')
        parser.add_argument('--downstream_patience', type=int, default=35, help='patience for downstream')
        
        return parent_parser

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train"
    
    def eval(self):
        super().eval()
        self.mode = "val"
    
    def on_test_start(self):
        self.mode = "test"

    def forward(self, x):
        if self.ae_net == 'ae':
            h_A, h_B, h_C = self.ae_model(x)
            if self.ds_latent_agg_method == 'concat':
                h = torch.cat([h_A, h_B, h_C], dim=1)
            elif self.ds_latent_agg_method == 'mean':
                h = torch.mean(torch.stack([h_A, h_B, h_C]), axis=0)
            elif self.ds_latent_agg_method == 'sum':
                h = torch.sum(torch.stack([h_A, h_B, h_C]), axis=0)
            elif self.ds_latent_agg_method == 'all':
                h = [h_A, h_B, h_C]
        elif self.ae_net == 'vae':
            h, _, _, _ = self.ae_model(x)
        return h

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
            output_dict['loss'] = output_dict['class_loss']
        elif self.ds_task == 'surv':
            output_dict = self.surv_step(batch)
            output_dict['loss'] = output_dict['surv_loss']
        elif self.ds_task == 'reg':
            output_dict = self.reg_step(batch)
            output_dict['loss'] = output_dict['reg_loss']
        elif self.ds_task == 'multi':
            output_dict = self.class_step(batch)
            output_dict.update(self.surv_step(batch))
            output_dict.update(self.reg_step(batch))
            output_dict['loss'] = output_dict['class_loss'] * self.ds_k_class + output_dict['surv_loss'] * self.ds_k_surv + output_dict['reg_loss'] * self.ds_k_reg
        return output_dict

    def class_step(self, batch):
        x_A, x_B, x_C = batch['x']
        y = batch['y']
        sample_ids = batch['sample_id']
        if self.ds_mask_A:
            x_A = self.mask_x_ch(x_A)
        if self.ds_mask_B:
            x_B = self.mask_x_ch(x_B)
        h = self.forward((x_A, x_B, x_C))
        if self.ds_add_omics_identity:
            down_loss = 0
            y_prob_omic = []
            for i, h_omic in enumerate(h):
                omic_id = torch.zeros((h_omic.shape[0], len(h))).to(self.device)
                omic_id[:, i] = 1
                h_omic = torch.cat([h_omic, omic_id], dim=1)
                y_out_omic = self.class_net(h_omic)
                if self.cl_loss == "wbce":
                    down_loss += self.wbce(y_out_omic, y, self.class_weights)
                elif self.cl_loss == "bce":
                    down_loss += self.criterion(y_out_omic, y)
                y_true = y.long()
                y_prob_omic.append(F.softmax(y_out_omic, dim=1))
            y_prob = torch.mean(torch.stack(y_prob_omic), axis=0)
            _, y_pred = torch.max(y_prob, 1)
            h = torch.cat(h, dim=1)
            
        else:
            y_out = self.class_net(h)
            if self.cl_loss == "wbce":
                down_loss = self.wbce(y_out, y, self.class_weights)
            elif self.cl_loss == "bce":
                down_loss = self.criterion(y_out, y)
            y_true = y.long()
            y_prob = F.softmax(y_out, dim=1)
            _, y_pred = torch.max(y_prob, 1)
            
        return {
            "class_loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true": y_true,
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach(),
            "y_out": y_out.detach()
        }

    def mask_x_ch(self, x):
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
        h = self.forward((x_A, x_B, x_C))
        y_out = self.surv_net(h)
        if self.survival_loss == 'MTLR':
            down_loss = MTLR_survival_loss(y_out, y_true, surv_E, self.tri_matrix_1)
        predict = self.predict_risk(y_out)
        survival = predict['survival']
        risk = predict['risk']
        return {
            "surv_loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true_E": surv_E.detach(),
            "y_true_T": surv_T.detach(),
            "survival": survival.detach(),
            "risk": risk.detach(),
            "y_out": y_out.detach()
        }
    
    def reg_step(self, batch):
        x_A, x_B, x_C = batch['x']
        v = batch['value']
        sample_ids = batch['sample_id']
        h = self.forward((x_A, x_B, x_C))
        v_bar = self.reg_net(h)
        loss = self.reg_loss(v_bar, v)
        return {
            "reg_loss": loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "v": v.detach(),
            "v_bar": v_bar.detach()
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
    
    def compute_reg_metrics(self, outputs):
        v = torch.cat([x["v"] for x in outputs]).cpu().numpy()
        v_bar = torch.cat([x["v_bar"] for x in outputs]).cpu().numpy()
        rmse =  sk.metrics.mean_squared_error(v, v_bar, squared=False)
        return rmse
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    def shared_epoch_end(self, outputs):
        if 'class' in self.ds_tasks:
            class_loss = torch.stack([x["class_loss"] for x in outputs]).mean()
            self.log("{}_class_loss".format(self.mode), class_loss)
            accuracy, precision, recall, f1, auc = self.compute_class_metrics(outputs)
            self.log("{}_accuracy".format(self.mode), accuracy)
            self.log("{}_precision".format(self.mode), precision)
            self.log("{}_recall".format(self.mode), recall)
            self.log("{}_f1".format(self.mode), f1)
            self.log("{}_auc".format(self.mode), auc)
        
        if 'surv' in self.ds_tasks:
            surv_loss = torch.stack([x["surv_loss"] for x in outputs]).mean()
            self.log("{}_surv_loss".format(self.mode), surv_loss)
            c_index, ibs = self.compute_surv_metrics(outputs)
            self.log("{}_c_index".format(self.mode), c_index)
            self.log("{}_ibs".format(self.mode), ibs)
        
        if 'reg' in self.ds_tasks:
            reg_loss = torch.stack([x["reg_loss"] for x in outputs]).mean()
            self.log("{}_reg_loss".format(self.mode), reg_loss)
            rmse = self.compute_reg_metrics(outputs)
            self.log("{}_rmse".format(self.mode), rmse)
        
        if self.ds_task == 'multi':
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log("{}_down_loss".format(self.mode), avg_loss)

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)
    
    def validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            if 'class' in self.ds_tasks:
                wandb.define_metric('val_accuracy', summary='max')
                wandb.define_metric('val_precision', summary='max')
                wandb.define_metric('val_recall', summary='max')
                wandb.define_metric('val_f1', summary='max')
                wandb.define_metric('val_auc', summary='max')
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)
    
    def test_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('test_accuracy', summary='max')
        return self.shared_step(batch)
    
    def test_epoch_end(self, outputs):
        self.shared_epoch_end(outputs)
    
    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def on_predict_epoch_end(self, results):
        outputs = results[0]
        if self.ds_task == 'class':
            y_true_binary = torch.cat([x["y_true"] for x in outputs]).cpu().numpy()
            y_true = np.argmax(y_true_binary, axis=1)
            y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().numpy()
            y_prob = torch.cat([x["y_prob"] for x in outputs]).cpu().numpy()
            y_out = torch.cat([x["y_out"] for x in outputs]).cpu().numpy()
            preds = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "y_true_binary": y_true_binary,
                "y_out": y_out
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
    

class ComicsModel(pl.LightningModule):
    def __init__(self, **config):
        super(ComicsModel, self).__init__()
        # super(ComicsModel, self).__init__(**config)
        # AutoEncoder.__init__(self, **config)
        # self.ae_model_path = config['ae_model_path']
        # self.ds_model_path = config['ds_model_path']
        # self.input_size_A = config['input_size_A']
        # self.input_size_B = config['input_size_B']
        # self.input_size_C = config['input_size_C']
        # self.batch_size = config['batch_size']
        # self.ae_net = config['ae_net']
        # self.ae_weight_kl = config['ae_weight_kl']
        # self.latent_size = config['latent_size']
        # self.ae_lr = config['ae_lr']
        # self.ae_weight_decay = config['ae_weight_decay']
        # self.ae_momentum = config['ae_momentum']
        # self.ae_drop_p = config['ae_drop_p']
        # self.ae_beta1 = config['ae_beta1']
        # self.ae_lr_policy = config['ae_lr_policy']
        # self.ae_epoch_num_decay = config['ae_epoch_num_decay']
        # self.ae_decay_step_size = config['ae_decay_step_size']
        # self.ae_max_epochs = config['pretraining_max_epochs']
        # self.cont_align_loss_weight = config['cont_align_loss_weight']
        # self.cont_noise_loss_weight = config['cont_noise_loss_weight']
        # self.cont_loss_temp = config['cont_loss_temp']
        # self.cont_loss_lambda = config['cont_loss_lambda']
        # self.add_distance_loss_to_latent = config['add_distance_loss_to_latent']
        # self.add_distance_loss_to_proj = config['add_distance_loss_to_proj']
        # self.distance_loss_weight = config['distance_loss_weight']
        # self.add_consistency_loss = config['add_consistency_loss']
        # self.consistency_loss_weight = config['consistency_loss_weight']
        # self.ae_optimizer = config['ae_optimizer']
        # self.ae_use_lrscheduler = config['ae_use_lrscheduler']
        # self.cont_align_loss_criterion = config['cont_align_loss_criterion']
        # self.cont_noise_loss_criterion = config['cont_noise_loss_criterion']
        # self.cont_align_loss_latent = config['cont_align_loss_latent']
        # self.cont_loss_similarity = config['cont_loss_similarity']
        # self.cont_loss_normalize = config['cont_loss_normalize']
        # self.cont_loss_p_norm = config['cont_loss_p_norm']
        # self.add_cont_type_loss = config['add_cont_type_loss']
        # self.split_A = config['split_A']
        # self.mask_A = config['mask_A']
        # self.num_mask_A = config['num_mask_A']
        # self.split_B = config['split_B']
        # self.mask_B = config['mask_B']
        # self.num_mask_B = config['num_mask_B']
        # self.mask_C = config['mask_C']
        # self.ratio_mask_C = config['ratio_mask_C']
        # self.masking_method = config['masking_method']
        # self.choose_masking_method_every_epoch = config['choose_masking_method_every_epoch']
        # self.use_one_encoder = config['use_one_encoder']
        # self.use_one_decoder = config['use_one_decoder']
        # self.concat_latent_for_decoder = config['concat_latent_for_decoder']
        # self.recon_all_thrice = config['recon_all_thrice']
        # self.predict_masked_chromosomes = config['predict_masked_chromosomes']
        # self.use_rep_trick = config['use_rep_trick']
        # if self.ae_net == "ae":
        #     if self.split_A and self.split_B:
        #         self.ae_dim_1B = 128
        #         self.ae_dim_1A = 128
        #         self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        #     elif self.split_A:
        #         self.ae_dim_1A = 128
        #         self.ae_dim_1B = 1024
        #         self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        #     elif self.split_B:
        #         self.ae_dim_1B = 128
        #         self.ae_dim_1A = 1024
        #         self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        # elif self.ae_net == "vae":
        #     if self.split_A and self.split_B:
        #         self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        #     elif self.split_B:
        #         self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        
        # self.projection_size = self.latent_size // 2
        # if self.cont_align_loss_criterion != "none":
        #     if self.cont_align_loss_criterion == "simclr":
        #         # self.cont_align_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
        #         self.projector = SimCLRProjectionHead(self.latent_size, self.latent_size, self.projection_size)
        #         self.cont_align_criterion = losses.NTXentLoss(temperature=self.cont_loss_temp)
        #     elif self.cont_align_loss_criterion == "clip":
        #         self.projector = CLIPProjectionHead(self.latent_size, self.projection_size, self.ae_drop_p)
        #         self.cont_align_criterion = CLIPLoss(temperature = self.cont_loss_temp, latent_size = self.latent_size, proj_size = self.projection_size)
        #     elif self.cont_align_loss_criterion == "barlowtwins":
        #         self.cont_align_criterion = BarlowTwinsLoss(lambd=self.cont_loss_lambda, latent_size=self.latent_size, proj_size=self.projection_size)
        #     elif self.cont_align_loss_criterion == 'ntxent':
        #         self.cont_align_criterion = NTXentLoss(latent_dim = self.latent_size, temperature=self.cont_loss_temp, batch_size=self.batch_size, similarity=self.cont_loss_similarity, normalize=self.cont_loss_normalize, p_norm=self.cont_loss_p_norm)
        #     elif self.cont_align_loss_criterion == 'simsiam':
        #         self.cont_align_criterion = SimSiamLoss(latent_dim=self.latent_size)
        
        # if self.cont_noise_loss_criterion != "none":
        #     if self.cont_noise_loss_criterion == "simclr":
        #         # self.cont_noise_criterion = SimCLR_Loss(batch_size = batch_size, temperature = self.cont_loss_temp, latent_size=self.latent_size, proj_size=self.projection_size)
        #         self.projector = SimCLRProjectionHead(self.latent_size, self.latent_size, self.projection_size)
        #         self.cont_noise_criterion = losses.NTXentLoss(temperature=self.cont_loss_temp)
        #     elif self.cont_noise_loss_criterion == "clip":
        #         self.projector = CLIPProjectionHead(self.latent_size, self.projection_size, self.ae_drop_p)
        #         self.cont_noise_criterion = CLIPLoss(temperature = self.cont_loss_temp, latent_size = self.latent_size, proj_size = self.projection_size)
        #     elif self.cont_noise_loss_criterion == "barlowtwins":
        #         self.cont_noise_criterion = BarlowTwinsLoss(lambd=self.cont_loss_lambda, latent_size=self.latent_size, proj_size=self.projection_size)
        #     elif self.cont_noise_loss_criterion == 'ntxent':
        #         self.cont_noise_criterion = NTXentLoss(latent_dim = self.latent_size, temperature=self.cont_loss_temp, batch_size=self.batch_size, similarity=self.cont_loss_similarity, normalize=self.cont_loss_normalize, p_norm=self.cont_loss_p_norm)
        #     elif self.cont_noise_loss_criterion == 'simsiam':
        #         self.cont_noise_criterion = SimSiamLoss(latent_dim=self.latent_size)
        
        # if self.add_distance_loss_to_latent or self.add_distance_loss_to_proj:
        #     self.distance_loss_criterion = config['distance_loss_criterion']
        #     if self.distance_loss_criterion == 'mse':
        #         self.dist_loss = nn.MSELoss()
        #     elif self.distance_loss_criterion == 'l1':
        #         self.dist_loss = nn.L1Loss()
        #     elif self.distance_loss_criterion == 'bce':
        #         self.dist_loss = nn.BCELoss()
        
        # if self.add_consistency_loss:
        #     self.cons_loss = nn.MSELoss()
        
        # if self.mask_B:
        #     self.mask_B_ids = np.random.randint(0, len(self.input_size_B), size=self.num_mask_B)
        # if self.mask_A:
        #     self.mask_A_ids = np.random.randint(0, len(self.input_size_A), size=self.num_mask_A)
        # if self.mask_C:
        #     self.mask_C_features = np.random.randint(0, self.input_size_C, size=int(self.ratio_mask_C * self.input_size_C))
        
        # self.change_ch_to_mask_every_epoch = config['change_ch_to_mask_every_epoch']

        # if self.predict_masked_chromosomes:
        #     num_mask_total = 1
        #     if self.split_A:
        #         num_mask_total += len(self.input_size_A)
        #     else:
        #         num_mask_total += 1
        #     if self.split_B:
        #         num_mask_total += len(self.input_size_B)
        #     else:
        #         num_mask_total += 1
        #     self.masked_chr_prediction_weight = config['masked_chr_prediction_weight']
        #     self.mask_pred_net = ClassifierNet(num_mask_total, self.latent_size * 3, dropout_p=ae_drop_p)
        #     self.mask_pred_criterion = nn.CrossEntropyLoss()

        # self.ds_input_size = self.latent_size
        # self.ds_drop_p = config['ds_drop_p']
        # self.num_classes = config['num_classes']
        # self.ds_lr = config['ds_lr']
        # self.ds_weight_decay = config['ds_weight_decay']
        # self.ds_beta1 = config['ds_beta1']
        # self.ds_lr_policy = config['ds_lr_policy']
        # self.ds_epoch_num_decay = config['ds_epoch_num_decay']
        # self.ds_decay_step_size = config['ds_decay_step_size']
        # self.class_weights = config['class_weights']
        # self.ae_net = config['ae_net']
        # self.ds_max_epochs = config['downstream_max_epochs']
        # self.ds_task = config['ds_task']
        # if self.ds_task == 'multi':
        #     self.ds_tasks = ['class', 'surv', 'reg']
        # else:
        #     self.ds_tasks = [self.ds_task]
        # self.ds_k_class = config['ds_k_class']
        # self.ds_k_surv = config['ds_k_surv']
        # self.ds_k_reg = config['ds_k_reg']
        # self.ds_save_latent_testing = config['ds_save_latent_testing']
        # self.ds_save_latent_training = config['ds_save_latent_training']
        # self.ds_mask_A = config['ds_mask_A']
        # self.ds_mask_B = config['ds_mask_B']
        # self.ds_masking_method = config['ds_masking_method']
        # # self.ae = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        # # if config['ds_freeze_ae'] == True:
        # #     self.ae.freeze()
        # self.ds_latent_agg_method = config['ds_latent_agg_method']
        # self.ds_add_omics_identity = config['ds_add_omics_identity']
        # if self.ds_add_omics_identity:
        #     self.ds_latent_agg_method = 'all'
        # self.ds_model = DownstreamModel.load_from_checkpoint(self.ds_model_path)
        # if 'class' in self.ds_tasks:
        #     self.class_net = self.ds_model.class_net
        #     self.wbce = weighted_binary_cross_entropy
        #     self.criterion = nn.CrossEntropyLoss()
        #     self.cl_loss = self.cl_loss
        # if 'surv' in self.ds_tasks:
        #     self.time_num = config['time_num']
        #     self.surv_net = self.ds_model.surv_net
        #     self.survival_loss = config["survival_loss"]
        #     if self.survival_loss == 'MTLR':
        #         self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
        #         self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)   
        #     self.survival_T_max = config['survival_T_max']
        # if 'reg' in self.ds_tasks:
        #     self.reg_net = self.ds_model.reg_net
        #     self.reg_loss = nn.MSELoss()

        # self.ae_model_path = config['ae_model_path']
        self.ds_model_path = config['ds_model_path']
        # self.ae_model = AutoEncoder.load_from_checkpoint(self.ae_model_path)
        # self.ae_model.unfreeze()
        # self.ae = self.ae_model.ae
        self.ds = DownstreamModel.load_from_checkpoint(self.ds_model_path)
        self.ae_model = self.ds.ae_model
        self.ae_model.unfreeze()
        # self.ae = self.ae_model.ae
        # if 'class' in self.ds.ds_tasks:
        #     self.class_net = self.ds.class_net
        # if 'surv' in self.ds.ds_tasks:
        #     self.surv_net = self.ds.surv_net
        # if 'reg' in self.ds.ds_tasks:
        #     self.reg_net = self.ds.reg_net
        self.cs_pretext_weight = config['cs_pretext_weight']
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ComicsModel")
        parser.add_argument("--cs_pretext_weight", type=float, default=1.0)
        return parent_parser
    
    def train(self, mode=True):
        super().train(mode)
        self.mode = "train"
    
    def eval(self):
        super().eval()
        self.mode = "val"
    
    def on_test_start(self):
        super().on_test_start()
        self.mode = "test"
    
    def configure_optimizers(self):
        optimizers, schedulers = [], []
        # if self.ae_optimizer == "adam":
        #     optimizers.append(optim.Adam(self.ae_model.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay))
        # if self.ae_optimizer == "lars":
        #     optimizers.append(LARS(self.ae_model.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay))
        # if self.ae_use_lrscheduler:
        #     schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizers[-1],
        #                                                     T_max=500,
        #                                                     eta_min=self.hparams.ae_lr/50))
        # optimizers.append(optim.Adam(self.ds.parameters(), lr=self.ds_lr, weight_decay=self.ds_weight_decay, betas=(self.ds_beta1, 0.999)))
        optimizers.append(optim.Adam(list(self.ds.parameters()) + list(self.ae_model.parameters()), lr=self.ds.ds_lr, weight_decay=self.ds.ds_weight_decay, betas=(self.ds.ds_beta1, 0.999)))
        if self.ds.ds_lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_lambda = 1.0 - max(0, epoch - self.ds.ds_max_epochs + self.ds.ds_epoch_num_decay) / float(self.ds.ds_epoch_num_decay + 1)
                return lr_lambda
            # lr_scheduler is imported from torch.optim
            schedulers.append(lr_scheduler.LambdaLR(optimizers[-1], lr_lambda=lambda_rule))
        elif self.ds.ds_lr_policy == 'step':
            schedulers.append(lr_scheduler.StepLR(optimizers[-1], step_size=self.ds.ds_decay_step_size, gamma=0.1))
        elif self.ds.ds_lr_policy == 'plateau':
            schedulers.append(lr_scheduler.ReduceLROnPlateau(optimizers[-1], mode='min', factor=0.2, threshold=0.01, patience=5))
        elif self.ds.ds_lr_policy == 'cosine':
            schedulers.append(lr_scheduler.CosineAnnealingLR(optimizers[-1], T_max=self.ds.ds_max_epochs, eta_min=0))
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        self.mode = "train"
        # if optimizer_idx == 0:
        #     pretext_loss = self.ae_model.training_step(batch, batch_idx)
        #     self.log('train_pretext_loss', pretext_loss, on_step=False, on_epoch=True)
        #     return pretext_loss
        # elif optimizer_idx == 1:
        #     ds_dict = self.ds.training_step(batch, batch_idx)
        #     self.log('train_class_loss', ds_dict['loss'], on_step=False, on_epoch=True)
        #     # total_loss = self.cs_pretext_weight * pretext_loss + ds_dict['loss']
        #     # self.log('train_comics_loss', total_loss, on_step=False, on_epoch=True)
        #     return {'loss': ds_dict['loss'], **ds_dict}

        pretext_loss = self.ae_model.training_step(batch, batch_idx)
        # self.log('train_pretext_loss', pretext_loss, on_step=False, on_epoch=True)
        ds_dict = self.ds.training_step(batch, batch_idx)
        # self.log('train_class_loss', ds_dict['loss'], on_step=False, on_epoch=True)
        total_loss = self.cs_pretext_weight * pretext_loss + ds_dict['loss']
        self.log('train_comics_loss', total_loss, on_step=False, on_epoch=True)
        return {'loss': total_loss, **ds_dict}
    
    def training_epoch_end(self, outputs):
        self.mode = 'train'
        self.ds.training_epoch_end(outputs)
    
    def validation_step(self, batch, batch_idx):
        self.mode = "val"
        ae_dict = self.ae_model.validation_step(batch, batch_idx)
        ds_dict = self.ds.validation_step(batch, batch_idx)
        pretext_loss = ae_dict['{}_pretext_loss'.format(self.mode)]
        # self.log('val_pretext_loss', pretext_loss, on_step=False, on_epoch=True)
        down_loss = 0
        if 'class' in self.ds.ds_tasks:
            down_loss += ds_dict['class_loss']
        if 'surv' in self.ds.ds_tasks:
            down_loss += ds_dict['surv_loss']
        if 'reg' in self.ds.ds_tasks:
            down_loss += ds_dict['reg_loss']
        # self.log('val_class_loss', down_loss, on_step=False, on_epoch=True)
        total_loss = self.cs_pretext_weight * pretext_loss + down_loss
        self.log('{}_comics_loss'.format(self.mode), total_loss, on_step=False, on_epoch=True)
        return {'ds_dict': ds_dict, 'ae_dict': ae_dict}

    def validation_epoch_end(self, outputs):
        self.mode = 'val'
        ae_outputs = [output['ae_dict'] for output in outputs]
        ds_outputs = [output['ds_dict'] for output in outputs]
        accuracy, _, _, _, _ = self.ds.compute_class_metrics(ds_outputs)
        self.log("val_accuracy", accuracy)
        self.ae_model.validation_epoch_end(ae_outputs)
        self.ds.validation_epoch_end(ds_outputs)
    
    def test_step(self, batch, batch_idx):
        self.mode = "test"
        ae_dict = self.ae_model.validation_step(batch, batch_idx)
        ds_dict = self.ds.test_step(batch, batch_idx)
        total_loss = self.cs_pretext_weight * ae_dict['val_pretext_loss']
        if 'class' in self.ds.ds_tasks:
            total_loss += ds_dict['class_loss']
        if 'surv' in self.ds.ds_tasks:
            total_loss += ds_dict['surv_loss']
        if 'reg' in self.ds.ds_tasks:
            total_loss += ds_dict['reg_loss']
        self.log('{}_comics_loss'.format(self.mode), total_loss, on_step=False, on_epoch=True)
        return {**ds_dict}

    def test_epoch_end(self, outputs):
        self.mode = 'test'
        self.ds.test_epoch_end(outputs)


class Comics(pl.LightningModule):
    def __init__(self, current_phase, **config):
        super(Comics, self).__init__()
        self.save_hyperparameters()
        if self.hparams.ae_net == "ae":
            if self.hparams.split_A and self.hparams.split_B:
                self.hparams.ae_dim_1B = 128
                self.hparams.ae_dim_1A = 128
                self.ae = AESepAB((self.hparams.input_size_A, self.hparams.input_size_B, self.hparams.input_size_C), self.hparams.latent_size, self.hparams.use_one_encoder, self.hparams.use_one_decoder, self.hparams.concat_latent_for_decoder, self.hparams.recon_all_thrice, self.hparams.use_rep_trick, dropout_p=self.hparams.ae_drop_p, dim_1B=self.hparams.ae_dim_1B, dim_2B=self.hparams.ae_dim_2B, dim_1A=self.hparams.ae_dim_1A, dim_2A=self.hparams.ae_dim_1A, dim_1C=self.hparams.ae_dim_1C, dim_2C=self.hparams.ae_dim_2C)
            elif self.hparams.split_A:
                self.hparams.ae_dim_1A = 128
                self.hparams.ae_dim_1B = 1024
                self.ae = AESepA((self.hparams.input_size_A, self.hparams.input_size_B, self.hparams.input_size_C), self.hparams.latent_size, self.hparams.use_one_decoder, self.hparams.concat_latent_for_decoder, self.hparams.recon_all_thrice, dropout_p=self.hparams.ae_drop_p, dim_1B=self.hparams.ae_dim_1B, dim_2B=self.hparams.ae_dim_2B, dim_1A=self.hparams.ae_dim_1A, dim_2A=self.hparams.ae_dim_2A, dim_1C=self.hparams.ae_dim_1C, dim_2C=self.hparams.ae_dim_2C)
            elif self.hparams.split_B:
                self.hparams.ae_dim_1B = 128
                self.hparams.ae_dim_1A = 1024
                self.ae = AESepB((self.hparams.input_size_A, self.hparams.input_size_B, self.hparams.input_size_C), self.hparams.latent_size, dropout_p=self.hparams.ae_drop_p, dim_1B=self.hparams.ae_dim_1B, dim_2B=self.hparams.ae_dim_2B, dim_1A=self.hparams.ae_dim_1A, dim_2A=self.hparams.ae_dim_2A, dim_1C=self.hparams.ae_dim_1C, dim_2C=self.hparams.ae_dim_2C)
        elif self.hparams.ae_net == "vae":
            if self.hparams.split_A and self.hparams.split_B:
                self.ae = VAESepAB((self.hparams.input_size_A, self.hparams.input_size_B, self.hparams.input_size_C), self.hparams.latent_size, dropout_p=self.hparams.ae_drop_p, dim_1B=self.hparams.ae_dim_1B, dim_2B=self.hparams.ae_dim_2B, dim_1A=self.hparams.ae_dim_1A, dim_2A=self.hparams.ae_dim_2A, dim_1C=self.hparams.ae_dim_1C, dim_2C=self.hparams.ae_dim_2C)
            elif self.hparams.split_B:
                self.ae = VAESepB((self.hparams.input_size_A, self.hparams.input_size_B, self.hparams.input_size_C), self.hparams.latent_size, dropout_p=self.hparams.ae_drop_p, dim_1B=self.hparams.ae_dim_1B, dim_2B=self.hparams.ae_dim_2B, dim_1A=self.hparams.ae_dim_1A, dim_2A=self.hparams.ae_dim_2A, dim_1C=self.hparams.ae_dim_1C, dim_2C=self.hparams.ae_dim_2C)
        
        self.hparams.projection_size = self.hparams.latent_size // 2
        if self.hparams.cont_align_loss_criterion != "none":
            if self.hparams.cont_align_loss_criterion == "simclr":
                # self.cont_align_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
                self.projector = SimCLRProjectionHead(self.hparams.latent_size, self.hparams.latent_size, self.hparams.projection_size)
                self.cont_align_criterion = losses.NTXentLoss(temperature=self.hparams.cont_loss_temp)
            elif self.hparams.cont_align_loss_criterion == "clip":
                self.projector = CLIPProjectionHead(self.hparams.latent_size, self.hparams.projection_size, self.hparams.ae_drop_p)
                self.cont_align_criterion = CLIPLoss(temperature = self.hparams.cont_loss_temp, latent_size = self.hparams.latent_size, proj_size = self.hparams.projection_size)
            elif self.hparams.cont_align_loss_criterion == "barlowtwins":
                self.cont_align_criterion = BarlowTwinsLoss(lambd=self.hparams.cont_loss_lambda, latent_size=self.hparams.latent_size, proj_size=self.hparams.projection_size)
            elif self.hparams.cont_align_loss_criterion == 'ntxent':
                self.cont_align_criterion = NTXentLoss(latent_dim = self.hparams.latent_size, temperature=self.hparams.cont_loss_temp, batch_size=self.hparams.batch_size, similarity=self.hparams.cont_loss_similarity, normalize=self.hparams.cont_loss_normalize, p_norm=self.hparams.cont_loss_p_norm)
            elif self.hparams.cont_align_loss_criterion == 'simsiam':
                self.cont_align_criterion = SimSiamLoss(latent_dim=self.hparams.latent_size)
        
        if self.hparams.cont_noise_loss_criterion != "none":
            if self.hparams.cont_noise_loss_criterion == "simclr":
                # self.cont_noise_criterion = SimCLR_Loss(batch_size = batch_size, temperature = cont_loss_temp, latent_size=latent_size, proj_size=self.projection_size)
                self.projector = SimCLRProjectionHead(self.hparams.latent_size, self.hparams.latent_size, self.hparams.projection_size)
                self.cont_noise_criterion = losses.NTXentLoss(temperature=self.hparams.cont_loss_temp)
            elif self.hparams.cont_noise_loss_criterion == "clip":
                self.projector = CLIPProjectionHead(self.hparams.latent_size, self.hparams.projection_size, self.hparams.ae_drop_p)
                self.cont_noise_criterion = CLIPLoss(temperature = self.hparams.cont_loss_temp, latent_size = self.hparams.latent_size, proj_size = self.hparams.projection_size)
            elif self.hparams.cont_noise_loss_criterion == "barlowtwins":
                self.cont_noise_criterion = BarlowTwinsLoss(lambd=self.hparams.cont_loss_lambda, latent_size=self.hparams.latent_size, proj_size=self.hparams.projection_size)
            elif self.hparams.cont_noise_loss_criterion == 'ntxent':
                self.cont_noise_criterion = NTXentLoss(latent_dim = self.hparams.latent_size, temperature=self.hparams.cont_loss_temp, batch_size=self.hparams.batch_size, similarity=self.hparams.cont_loss_similarity, normalize=self.hparams.cont_loss_normalize, p_norm=self.hparams.cont_loss_p_norm)
            elif self.hparams.cont_noise_loss_criterion == 'simsiam':
                self.cont_noise_criterion = SimSiamLoss(latent_dim=self.hparams.latent_size)
        
        if self.hparams.add_distance_loss_to_latent or self.hparams.add_distance_loss_to_proj:
            if self.hparams.distance_loss_criterion == 'mse':
                self.dist_loss = nn.MSELoss()
            elif self.hparams.distance_loss_criterion == 'l1':
                self.dist_loss = nn.L1Loss()
            elif self.hparams.distance_loss_criterion == 'bce':
                self.dist_loss = nn.BCELoss()
        
        if self.hparams.add_consistency_loss:
            self.cons_loss = nn.MSELoss()
        
        if self.hparams.mask_B:
            self.mask_B_ids = np.random.randint(0, len(self.hparams.input_size_B), size=self.hparams.num_mask_B)
        if self.hparams.mask_A:
            self.mask_A_ids = np.random.randint(0, len(self.hparams.input_size_A), size=self.hparams.num_mask_A)
        if self.hparams.mask_C:
            self.mask_C_features = np.random.randint(0, self.hparams.input_size_C, size=int(self.hparams.ratio_mask_C * self.hparams.input_size_C))

        if self.hparams.predict_masked_chromosomes:
            num_mask_total = 1
            if self.hparams.split_A:
                num_mask_total += len(self.hparams.input_size_A)
            else:
                num_mask_total += 1
            if self.hparams.split_B:
                num_mask_total += len(self.hparams.input_size_B)
            else:
                num_mask_total += 1
            self.mask_pred_net = ClassifierNet(num_mask_total, self.hparams.latent_size * 3, dropout_p=self.hparams.ae_drop_p)
            self.mask_pred_criterion = nn.CrossEntropyLoss()

        ########################################################################################################################
        self.hparams.ds_input_size = config['latent_size']
        if self.hparams.ds_task == 'multi':
            self.ds_tasks = ['class', 'surv', 'reg']
        else:
            self.ds_tasks = [self.hparams.ds_task]
        if self.hparams.ds_add_omics_identity:
            self.hparams.ds_latent_agg_method = 'all'
        if 'class' in self.ds_tasks:
            self.class_net = ClassifierNet(self.hparams.num_classes, self.hparams.ds_input_size, dropout_p=self.hparams.ds_drop_p)
            self.wbce = weighted_binary_cross_entropy
            self.criterion = nn.CrossEntropyLoss()
            self.cl_loss = self.hparams.cl_loss
        if 'surv' in self.ds_tasks:
            self.surv_net = SurvivalNet(self.hparams.time_num, self.hparams.ds_input_size, dropout_p=self.hparams.ds_drop_p)
            if self.hparams.survival_loss == 'MTLR':
                self.tri_matrix_1 = self.get_tri_matrix(dimension_type=1)
                self.tri_matrix_2 = self.get_tri_matrix(dimension_type=2)   
        if 'reg' in self.ds_tasks:
            self.reg_net = RegressionNet(self.hparams.ds_input_size, dropout_p=self.hparams.ds_drop_p)
            self.reg_loss = nn.MSELoss()

        ##########################################################################################################################
        self.hparams.current_phase = current_phase
        self.hparams.lr = config['cs_{}_lr'.format(self.hparams.current_phase)]
        self.hparams.weight_decay = config['cs_{}_weight_decay'.format(self.hparams.current_phase)]
        self.hparams.max_epochs = config['cs_{}_max_epochs'.format(self.hparams.current_phase)]
        self.hparams.lr_policy = config['cs_{}_lr_policy'.format(self.hparams.current_phase)]
        if self.hparams.current_phase == 'p1':
            self.set_ae_requires_grad(True)
            self.set_ds_requires_grad(False)
        elif self.hparams.current_phase == 'p2':
            self.set_ae_requires_grad(False)
            self.set_ds_requires_grad(True)
        elif self.hparams.current_phase == 'p3':
            self.set_ae_requires_grad(True)
            self.set_ds_requires_grad(True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Comics")
        parser.add_argument("--ae_net", type=str, default="vae",
                            help="AutoEncoder network architecture, options: [ae, vae]")
        parser.add_argument("--ae_weight_kl", type=float, default=0.01,
                            help="Weight for KL loss if vae is used")
        parser.add_argument("--latent_size", type=int, default=512)
        parser.add_argument("--projection_size", type=int, default=256)
        parser.add_argument("--ae_drop_p", type=float, default=0.2)
        parser.add_argument("--cont_align_loss_criterion", type=str, default="barlowtwins", help="contrastive alignment loss to use, options: none, simclr, clip, barlowtwins, ntxent, simsiam")
        parser.add_argument("--cont_noise_loss_criterion", type=str, default="barlowtwins", help="contrastive noise loss to use, options: none, clip, barlowtwins, ntxent")
        parser.add_argument("--add_cont_type_loss", default=False, type=lambda x: (str(x).lower() == 'true')
                            , help="Add cancer type wise contrastive loss")
        parser.add_argument("--cont_loss_similarity", type=str, default="cosine", help="similarity function to use for ntxent loss, options: [cosine, dot]")
        parser.add_argument("--cont_loss_normalize", default=False, type=lambda x: (str(x).lower() == 'true'), help="whether to normalize ntxent loss")
        parser.add_argument("--cont_loss_p_norm", type=float, default=2.0, help="p-norm to use for ntxent loss if cont_loss_normalize is set to true")
        parser.add_argument("--cont_loss_temp", type=float, default=0.1)
        parser.add_argument("--cont_loss_lambda", type=float, default=0.0051, help="for barlowtwins")
        parser.add_argument("--cont_align_loss_weight", type=float, default=0.5)
        parser.add_argument("--cont_noise_loss_weight", type=float, default=0.5)
        parser.add_argument("--cont_align_loss_latent", type=str, default="masked", help="the latent representation on which contrastive alignment loss should be computed, options=['masked', 'unmasked', 'mean']")
        parser.add_argument("--add_distance_loss_to_latent", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--add_distance_loss_to_proj", default=False, type=lambda x: (str(x).lower() == 'true'), help="works only when a constrastive loss is used")
        parser.add_argument("--distance_loss_weight", type=float, default=0.5)
        parser.add_argument("--distance_loss_criterion", type=str, default="mse", help="distance loss to use, options: mse, bce, l1")
        parser.add_argument("--add_consistency_loss", default=False, type=lambda x: (str(x).lower() == 'true'), help="add consistency loss")
        parser.add_argument("--consistency_loss_weight", type=float, default=10.0)
        parser.add_argument("--ae_dim_1B", type=int, default=128)
        parser.add_argument("--ae_dim_2B", type=int, default=1024)
        parser.add_argument("--ae_dim_1A", type=int, default=2048)
        parser.add_argument("--ae_dim_2A", type=int, default=1024)
        parser.add_argument("--ae_dim_1C", type=int, default=1024)
        parser.add_argument("--ae_dim_2C", type=int, default=1024)
        parser.add_argument('--mask_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, num_mask_A chromosomes of A are masked')
        parser.add_argument('--num_mask_A', type=int, default=0,
                                help='number of chromosomes of A to mask')
        parser.add_argument('--mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, num_mask_B chromosomes of B are masked')
        parser.add_argument('--num_mask_B', type=int, default=0,
                                help='number of chromosomes of B to mask')
        parser.add_argument('--mask_C', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, ratio_mask_C of C are masked')
        parser.add_argument('--ratio_mask_C', type=float, default=0.0,
                                help='ratio of C to mask')
        parser.add_argument('--masking_method', type=str, default='zero',
                                help='method to mask data, can be "zero", "gaussian_noise", or "swap_noise"')
        parser.add_argument('--choose_masking_method_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the masking method is chosen randomly each epoch and "masking_method" argument is ignored')
        parser.add_argument('--change_ch_to_mask_every_epoch', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the chromosomes to mask are changed each epoch')
        parser.add_argument('--predict_masked_chromosomes', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, the chromosomes that are masked will be predicted')
        parser.add_argument('--masked_chr_prediction_weight', type=float, default=0.5,
                                help='weight of masked chromosomes prediction loss')
        parser.add_argument('--recon_all_thrice', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, modalities A, B and C will be reconstructed from latent representations of each A, B and C modalities')
        parser.add_argument('--use_one_encoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, only one encoder is used to represent all modalities')
        parser.add_argument('--use_one_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, only one decoder is used to reconstruct all modalities')
        parser.add_argument('--concat_latent_for_decoder', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, latent vectors from A, B and C are concatenated before being fed into the decoder')
        parser.add_argument('--use_rep_trick', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='use reparameterization in trick in ae')
        parser.add_argument('--add_MMD_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add MMD loss')
        parser.add_argument('--MMD_loss_weight', type=float, default=0.5,
                                help='weight of MMD loss')
        parser.add_argument('--add_latent_reconstruction_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add latent reconstruction loss')
        parser.add_argument('--latent_reconstruction_loss_weight', type=float, default=0.5, 
                                help='weight of latent reconstruction loss')
        ################################################################################################################
        parser.add_argument("--ds_drop_p", type=float, default=0.2)
        parser.add_argument("--num_classes", type=int, default=34)
        parser.add_argument("--cl_loss", type=str, default="wbce", help="Loss function to use. Options: wbce, bce")
        parser.add_argument("--ds_task", type=str, default='class', 
                            help='downstream task, options: class (classification like cancer type classification), surv (survival analysis), reg (regression like age prediction), multi (multi-task training of all 3 tasks together)')
        parser.add_argument("--ds_k_class", type=float, default=1.0, help="Weight for classification loss in multi-task training")
        parser.add_argument("--ds_k_surv", type=float, default=1.0, help="Weight for survival loss in multi-task training")
        parser.add_argument("--ds_k_reg", type=float, default=1.0, help="Weight for regression loss in multi-task training")
        parser.add_argument('--survival_loss', type=str, default='MTLR', help='choose the survival loss')
        parser.add_argument('--survival_T_max', type=float, default=-1, help='maximum T value for survival prediction task')
        parser.add_argument('--time_num', type=int, default=256, help='number of time intervals in the survival model')
        parser.add_argument('--ds_latent_agg_method', type=str, default='mean',
                                help='method to aggregate latent representations from autoencoders of A, B and C, options: "mean", "concat", "sum", "all" (pass all latents one by one)')
        parser.add_argument('--ds_save_latent_testing', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of testing data')
        parser.add_argument('--ds_save_latent_training', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of training data')
        parser.add_argument('--ds_save_latent_val', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of validation data')
        parser.add_argument('--ds_save_latent_dataset', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='whether to save the latent representations of the whole dataset')
        parser.add_argument('--ds_mask_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from A will be masked')
        parser.add_argument('--ds_mask_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, data from B will be masked')
        parser.add_argument('--ds_masking_method', type=str, default='zero',
                                help='method to mask A, options: "zero", "gaussian_noise"')
        parser.add_argument('--ds_class_callback_key', type=str, default='accuracy',
                                help='key for the callback to use for classification task')
        parser.add_argument('--ds_surv_callback_key', type=str, default='c_index',
                                help='key for the callback to use for survival task')
        parser.add_argument('--ds_reg_callback_key', type=str, default='mse',
                                help='key for the callback to use for regression task')
        parser.add_argument('--ds_add_omics_identity', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='add omics id to latent representations before using them for classification task')
        ################################################################################################################
        parser.add_argument('--cs_pretext_weight', type=float, default=1.0, help='weight for pretext loss in total comics loss')
        parser.add_argument('--cs_p1_max_epochs', type=int, default=50, help='maximum number of epochs for phase 1 (only pretext loss is minimised)')
        parser.add_argument('--cs_p1_patience', type=int, default=35, help='patience for phase 1')
        parser.add_argument('--cs_p1_lr', type=float, default=1e-3, help='learning rate for phase 1')
        parser.add_argument('--cs_p1_weight_decay', type=float, default=1e-4, help='weight decay for phase 1')
        parser.add_argument('--cs_p1_lr_policy', type=str, default='none',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine | none]')
        parser.add_argument('--cs_p2_max_epochs', type=int, default=50, help='maximum number of epochs for phase 2 (only downstream loss is minimised)')
        parser.add_argument('--cs_p2_patience', type=int, default=15, help='patience for phase 2')
        parser.add_argument('--cs_p2_lr', type=float, default=1e-4, help='learning rate for phase 2')
        parser.add_argument('--cs_p2_weight_decay', type=float, default=1e-4, help='weight decay for phase 2')
        parser.add_argument('--cs_p2_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine | none]')
        parser.add_argument('--cs_p3_max_epochs', type=int, default=150, help='maximum number of epochs for phase 3 (total loss is minimised)')
        parser.add_argument('--cs_p3_patience', type=int, default=35, help='patience for phase 3')
        parser.add_argument('--cs_p3_lr', type=float, default=1e-4, help='learning rate for phase 3')
        parser.add_argument('--cs_p3_weight_decay', type=float, default=1e-4, help='weight decay for phase 3')
        parser.add_argument('--cs_p3_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine | none]')
        parser.add_argument("--resume_training", default=False, type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument("--model_path", type=str, default="")
        parser.add_argument("--cs_beta1", type=float, default=0.5)
        parser.add_argument('--cs_lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine | none]')
        parser.add_argument('--cs_epoch_num_decay', type=int, default=50,
                            help='Number of epoch to linearly decay learning rate to zero (lr_policy == linear)')
        parser.add_argument('--cs_decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch (lr_policy == step)')
        parser.add_argument("--cs_optimizer", type=str, default="adam", help="optimizer to use, options: adam, lars")
        return parent_parser    

    def set_ae_requires_grad(self, requires_grad=False):
        for param in self.ae.parameters():
                param.requires_grad = requires_grad
        if self.hparams.cont_noise_loss_criterion != "none":
            for param in self.cont_noise_criterion.parameters():
                param.requires_grad = requires_grad
        if self.hparams.cont_align_loss_criterion != "none":
            for param in self.cont_align_criterion.parameters():
                param.requires_grad = requires_grad
    
    def set_ds_requires_grad(self, requires_grad=False):
        if 'class' in self.ds_tasks:
            for param in self.class_net.parameters():
                param.requires_grad = requires_grad
        if 'survival' in self.ds_tasks:
            for param in self.surv_net.parameters():
                param.requires_grad = requires_grad
        if 'regression' in self.ds_tasks:
            for param in self.reg_net.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):
        if self.hparams.ae_net == 'ae':
            h_A, h_B, h_C = self.ae(x)
            if self.hparams.ds_latent_agg_method == 'concat':
                h = torch.cat([h_A, h_B, h_C], dim=1)
            elif self.hparams.ds_latent_agg_method == 'mean':
                h = torch.mean(torch.stack([h_A, h_B, h_C]), axis=0)
            elif self.hparams.ds_latent_agg_method == 'sum':
                h = torch.sum(torch.stack([h_A, h_B, h_C]), axis=0)
            elif self.hparams.ds_latent_agg_method == 'all':
                h = [h_A, h_B, h_C]
        elif self.hparams.ae_net == 'vae':
            h, _, _, _ = self.ae(x)
        return h

    def configure_optimizers(self):
        if self.hparams.cs_optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.cs_optimizer == "lars":
            optimizer = LARS(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_lambda = 1.0 - max(0, epoch - self.hparams.max_epochs + self.hparams.cs_epoch_num_decay) / float(self.hparams.cs_epoch_num_decay + 1)
                return lr_lambda
            # lr_scheduler is imported from torch.optim
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.hparams.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.hparams.cs_decay_step_size, gamma=0.1)
        elif self.hparams.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif self.hparams.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=0)
        elif self.hparams.lr_policy == 'none':
            return optimizer
        return [optimizer], [scheduler]
    
    def train(self, mode=True):
        super().train(mode)
        self.mode = "train"
    
    def eval(self):
        super().eval()
        self.mode = "val"

    def on_test_start(self):
        self.mode = "test"
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.hparams.change_ch_to_mask_every_epoch:
            if self.hparams.mask_B:
                self.mask_B_ids = np.random.randint(0, len(self.hparams.input_size_B), size=self.hparams.num_mask_B)
            if self.hparams.mask_A:
                self.mask_A_ids = np.random.randint(0, len(self.hparams.input_size_A), size=self.hparams.num_mask_A)
            if self.hparams.mask_C:
                self.mask_C_features = np.random.randint(0, self.hparams.input_size_C, size=int(self.hparams.ratio_mask_C * self.hparams.input_size_C))
        if self.hparams.choose_masking_method_every_epoch:
            self.hparams.masking_method = np.random.choice(['zero', 'gaussian_noise', 'swap_noise'])

    def mask_x_ch(self, x, mask_ids):
        x_masked = []
        for i in range(len(x)):
            x_masked.append(x[i])
            if i in mask_ids:
                if self.hparams.masking_method == 'zero':
                    x_masked[-1] = torch.zeros_like(x_masked[-1])
                elif self.hparams.masking_method == 'gaussian_noise':
                    x_masked[-1] = x_masked[-1] + torch.randn_like(x_masked[-1])
                elif self.hparams.masking_method == 'swap_noise':
                    for j in range(x_masked[-1].shape[1]):
                        x_masked[-1][:, j] = x_masked[-1][torch.randperm(x_masked[-1].shape[0]), j]
        return x_masked
    
    def mask_x_feat(self, x, mask_feats):
        if self.hparams.masking_method == 'zero':
            x[:, mask_feats] = torch.zeros_like(x[:, mask_feats])
        elif self.hparams.masking_method == 'gaussian_noise':
            x[:, mask_feats] = x[:, mask_feats] + torch.randn_like(x[:, mask_feats])
        elif self.hparams.masking_method == 'swap_noise':
            for j in range(x.shape[1]):
                x[:, j] = x[torch.randperm(x.shape[0]), j]
        return x
    
    def sum_subset_losses(self, x_recon, x):
        x_recon_loss = []
        for i in range(len(x)):
            x_recon_loss.append(F.mse_loss(x_recon[i], x[i]))
        return sum(x_recon_loss)
    
    def sum_losses(self, x_A_recon, x_B_recon, x_C_recon, x_A, x_B, x_C):
        if self.hparams.split_A:
            recon_A_loss = self.sum_subset_losses(x_A_recon, x_A)
        else:
            recon_A_loss = F.mse_loss(x_A_recon, x_A)
        if self.hparams.split_B:
            recon_B_loss = self.sum_subset_losses(x_B_recon, x_B)
        else:
            recon_B_loss = F.mse_loss(x_B_recon, x_B)
        recon_C_loss = F.mse_loss(x_C_recon, x_C)
        if self.hparams.mask_A and self.hparams.mask_B:
            recon_loss = 0.4 * recon_A_loss + 0.4 * recon_B_loss + 0.2 * recon_C_loss
        elif self.hparams.mask_A:
            recon_loss = 0.5 * recon_A_loss + 0.25 * recon_B_loss + 0.25 * recon_C_loss 
        elif self.hparams.mask_B:
            recon_loss = 0.5 * recon_B_loss + 0.25 * recon_A_loss + 0.25 * recon_C_loss
        else:
            recon_loss = recon_A_loss + recon_B_loss + recon_C_loss
        return recon_loss

    def ae_step(self, batch):
        logs = {}
        pretext_loss = 0
        x_A, x_B, x_C = batch['x']
        x_A_in, x_B_in, x_C_in = x_A, x_B, x_C
        if self.hparams.mask_A:
            x_A_in = self.mask_x_ch(x_A, self.mask_A_ids)
        if self.hparams.mask_B:
            x_B_in = self.mask_x_ch(x_B, self.mask_B_ids)
        if self.hparams.mask_C:
            x_C_in = self.mask_x_feat(x_C, self.mask_C_features)
        h_A, h_B, h_C = self.ae.encode((x_A_in, x_B_in, x_C_in))
        h_A_unmasked, h_B_unmasked, h_C_unmasked = self.ae.encode((x_A, x_B, x_C))
        if self.hparams.cont_noise_loss_criterion != "none":
            cont_noise_loss = 0
            for omics_type, h_masked, h_unmasked in zip(['A', 'B', 'C'], [h_A, h_B, h_C], [h_A_unmasked, h_B_unmasked, h_C_unmasked]):
                cont_noise_logs_type, cont_noise_loss_type = self.cont_noise_step(h_masked, h_unmasked, omics_type)
                logs.update(cont_noise_logs_type)
                cont_noise_loss += cont_noise_loss_type
            logs['{}_cont_noise_loss_total'.format(self.mode)] = cont_noise_loss
            pretext_loss += (cont_noise_loss / 3) * self.hparams.cont_noise_loss_weight
        if self.hparams.predict_masked_chromosomes:
            h = torch.cat((h_A, h_B, h_C), dim=1)
            mask_y_out = self.mask_pred_net(h)
            mask_y = torch.zeros(mask_y_out.shape).to(self.device)
            if self.hparams.mask_A:
                mask_y[:, self.mask_A_ids] = 1
            if self.hparams.mask_B:
                mask_y[:, self.mask_B_ids] = 1
            mask_pred_loss = self.mask_pred_criterion(mask_y_out, mask_y)
            pretext_loss += self.hparams.masked_chr_prediction_weight * mask_pred_loss
            logs['{}_mask_pred_loss'.format(self.mode)] = mask_pred_loss
        if self.hparams.recon_all_thrice:
            recon_list = self.ae.decode((h_A, h_B, h_C))
        else:
            recon_list.append(self.ae.decode((h_A, h_B, h_C)))
        recon_loss = 0
        for i, x_recon in enumerate(recon_list):
            x_A_recon, x_B_recon, x_C_recon = x_recon
            recon_loss_all = self.sum_losses(x_A_recon, x_B_recon, x_C_recon, x_A, x_B, x_C)
            recon_loss += recon_loss_all
            if self.hparams.recon_all_thrice:
                logs['{}_recon_all_from_{}_loss'.format(self.mode, string.ascii_uppercase[i])] = recon_loss_all
            else:
                logs['{}_recon_all_loss'.format(self.mode)] = recon_loss_all
        pretext_loss += recon_loss
        if self.hparams.recon_all_thrice:
            logs['{}_total_recon_all_loss'.format(self.mode)] = recon_loss
        
        if self.hparams.cont_align_loss_latent == 'unmasked':
            h_A, h_B, h_C = h_A_unmasked, h_B_unmasked, h_C_unmasked
        elif self.hparams.cont_align_loss_latent == 'mean':
            h_A, h_B, h_C = (h_A + h_A_unmasked) / 2, (h_B + h_B_unmasked) / 2, (h_C + h_C_unmasked) / 2
        
        return logs, (h_A, h_B, h_C), pretext_loss 
    
    def vae_step(self, batch):
        logs = {}
        x_A, x_B, x_C = batch['x']
        if self.hparams.mask_B:
            x_B_masked = self.mask_x_ch(x_B, self.mask_B_ids)
            z, recon_x, mean, log_var = self.ae((x_A, x_B_masked, x_C))
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = F.mse_loss(recon_x[0], x_A) + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_B_loss + kl_loss
            logs['{}_recon_B_loss'.format(self.mode)] = recon_B_loss
            logs['{}_recon_B_kl_loss'.format(self.mode)] = recon_loss
        
        elif self.hparams.mask_A:
            x_A_masked = self.mask_x_ch(x_A, self.mask_A_ids)
            h, recon_x, mean, log_var = self.ae((x_A_masked, x_B, x_C))
            recon_A_loss = self.sum_subset_losses(recon_x[0], x_A)
            recon_B_loss = self.sum_subset_losses(recon_x[1], x_B)
            recon_loss_all = recon_A_loss + recon_B_loss + F.mse_loss(recon_x[2], x_C)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            recon_loss = recon_A_loss + kl_loss
            logs['{}_recon_A_loss'.format(self.mode)] = recon_A_loss
            logs['{}_recon_A_kl_loss'.format(self.mode)] = recon_loss

        else:
            h, recon_x, mean, log_var = self.ae((x_A, x_B, x_C))
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
    
    def cont_align_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        if self.hparams.cont_align_loss_criterion == "clip":
            loss_A_B, loss_A1, loss_B1 = self.cont_align_criterion(h_A, h_B)
            loss_B_C, loss_B2, loss_C1 = self.cont_align_criterion(h_B, h_C)
            loss_C_A, loss_C2, loss_A2 = self.cont_align_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_A = (loss_A1 + loss_A2) / 2
            loss_B = (loss_B1 + loss_B2) / 2
            loss_C = (loss_C1 + loss_C2) / 2
            logs['{}_cont_{}_loss_A'.format(self.mode, self.cont_pair)] = loss_A
            logs['{}_cont_{}_loss_B'.format(self.mode, self.cont_pair)] = loss_B
            logs['{}_cont_{}_loss_C'.format(self.mode, self.cont_pair)] = loss_C

        elif self.hparams.cont_align_loss_criterion == "barlowtwins":
            loss_A_B, loss_on_diag1, loss_off_diag1, z_A, z_B = self.cont_align_criterion(h_A, h_B)
            loss_B_C, loss_on_diag2, loss_off_diag2, _, z_C = self.cont_align_criterion(h_B, h_C)
            loss_C_A, loss_on_diag3, loss_off_diag3, _, _ = self.cont_align_criterion(h_C, h_A)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_on_diag = (loss_on_diag1 + loss_on_diag2 + loss_on_diag3) / 3
            loss_off_diag = (loss_off_diag1 + loss_off_diag2 + loss_off_diag3) / 3
            logs['{}_cont_{}_loss_on_diag'.format(self.mode, self.cont_pair)] = loss_on_diag
            logs['{}_cont_{}_loss_off_diag'.format(self.mode, self.cont_pair)] = loss_off_diag
        
        elif self.hparams.cont_align_loss_criterion == "simclr":
            z_A = self.projector(h_A)
            z_B = self.projector(h_B)
            z_C = self.projector(h_C)
            z_AB = torch.cat((z_A, z_B), dim=0)
            z_BC = torch.cat((z_B, z_C), dim=0)
            z_CA = torch.cat((z_C, z_A), dim=0)
            labels = torch.arange(z_A.shape[0]).repeat(2)
            loss_A_B, loss_num1, loss_den1 = self.cont_align_criterion(z_AB, labels)
            loss_B_C, loss_num2, loss_den2 = self.cont_align_criterion(z_BC, labels)
            loss_C_A, loss_num3, loss_den3 = self.cont_align_criterion(z_CA, labels)
            cont_loss = (loss_A_B + loss_B_C + loss_C_A) / 3
            loss_num = (loss_num1 + loss_num2 + loss_num3) / 3
            loss_den = (loss_den1 + loss_den2 + loss_den3) / 3
            logs['{}_cont_{}_loss_num'.format(self.mode, self.cont_pair)] = loss_num
            logs['{}_cont_{}_loss_den'.format(self.mode, self.cont_pair)] = loss_den
        
        elif self.hparams.cont_align_loss_criterion == "ntxent":
            loss_A_B, z_A, z_B = self.cont_align_criterion(h_A, h_B)
            loss_B_C, _, z_C = self.cont_align_criterion(h_B, h_C)
            loss_C_A, _, _ = self.cont_align_criterion(h_C, h_A)
            cont_loss = loss_A_B + loss_B_C + loss_C_A
        
        elif self.hparams.cont_align_loss_criterion == "simsiam":
            cont_loss, z_A, z_B, z_C = self.cont_align_criterion(h_A, h_B, h_C)

        logs['{}_cont_{}_loss'.format(self.mode, self.cont_pair)] = cont_loss
        if self.hparams.add_distance_loss_to_proj:
            _, dist_loss = self.dist_step((z_A, z_B, z_C))
            cont_loss += dist_loss * self.hparams.distance_loss_weight
            self.log('{}_dist_loss_btw_proj'.format(self.mode), dist_loss, on_step=False, on_epoch=True)
        return logs, cont_loss
    
    def cont_noise_step(self, h_masked, h_unmasked, omics_type='A'):
        logs = {}
        if self.hparams.cont_noise_loss_criterion == "clip":
            cont_noise_loss, loss_masked, loss_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
            # logs['{}_cont_noise_loss_{}_masked'.format(self.mode, omics_type)] = loss_masked
            # logs['{}_cont_noise_loss_{}_unmasked'.format(self.mode, omics_type)] = loss_unmasked
        elif self.hparams.cont_noise_loss_criterion == "barlowtwins":
            cont_noise_loss, loss_on_diag, loss_off_diag, z_masked, z_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
            # logs['{}_cont_noise_{}_loss_on_diag'.format(self.mode, omics_type)] = loss_on_diag
            # logs['{}_cont_noise_{}_loss_off_diag'.format(self.mode, omics_type)] = loss_off_diag
        elif self.hparams.cont_noise_loss_criterion == "ntxent":
            cont_noise_loss, z_masked, z_unmasked = self.cont_noise_criterion(h_masked, h_unmasked)
        logs['{}_cont_noise_loss_{}'.format(self.mode, omics_type)] = cont_noise_loss
        return logs, cont_noise_loss
        
    def dist_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        if self.hparams.distance_loss_criterion == 'bce':
            h_A = torch.clamp(h_A, min=1e-7, max=0.9999)
            h_B = torch.clamp(h_B, min=1e-7, max=0.9999)
            h_C = torch.clamp(h_C, min=1e-7, max=0.9999)
        dist_loss = self.dist_loss(h_A, h_B) + self.dist_loss(h_B, h_C) + self.dist_loss(h_C, h_A)
        return logs, dist_loss
    
    def cons_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        cons_loss = 0
        h_A_recon_using_dec_B = self.ae.encode_B(self.ae.decode_h_B(h_A)[1])
        h_B_recon_using_dec_A = self.ae.encode_A(self.ae.decode_h_A(h_B)[0])
        h_C_recon_using_dec_A = self.ae.encode_A(self.ae.decode_h_A(h_C)[0])
        h_A_recon_using_dec_C = self.ae.encode_C(self.ae.decode_h_C(h_A)[2])
        h_B_recon_using_dec_C = self.ae.encode_C(self.ae.decode_h_C(h_B)[2])
        h_C_recon_using_dec_B = self.ae.encode_B(self.ae.decode_h_B(h_C)[1])
        cons_loss += self.cons_loss(h_A_recon_using_dec_B, h_B)
        cons_loss += self.cons_loss(h_B_recon_using_dec_A, h_A)
        cons_loss += self.cons_loss(h_C_recon_using_dec_A, h_A)
        cons_loss += self.cons_loss(h_A_recon_using_dec_C, h_C)
        cons_loss += self.cons_loss(h_B_recon_using_dec_C, h_C)
        cons_loss += self.cons_loss(h_C_recon_using_dec_B, h_B)
        return logs, cons_loss
    
    def MMD_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        MMD_loss = 0
        MMD_loss += mmd_rbf(h_A, h_B)
        MMD_loss += mmd_rbf(h_B, h_C)
        MMD_loss += mmd_rbf(h_C, h_A)
        return logs, MMD_loss
    
    def latent_recon_step(self, h):
        logs = {}
        h_A, h_B, h_C = h
        latent_recon_loss = 0
        if self.hparams.recon_all_thrice:
            recon_list = self.ae.decode((h_A, h_B, h_C))
            x_recon = (recon_list[0][0], recon_list[1][1], recon_list[2][2])
            h_A_recon, h_B_recon, h_C_recon = self.ae.encode(x_recon)
        latent_recon_loss += F.mse_loss(h_A, h_A_recon)
        latent_recon_loss += F.mse_loss(h_B, h_B_recon)
        latent_recon_loss += F.mse_loss(h_C, h_C_recon)
        return logs, latent_recon_loss

    def ae_training_step(self, batch, batch_idx):
        if self.hparams.ae_net == 'ae':
            logs, h, pretext_loss = self.ae_step(batch) 
        elif self.hparams.ae_net == 'vae':
            logs, h, pretext_loss = self.vae_step(batch)
        for k, v in logs.items():
            self.log(k, v, on_step=False, on_epoch=True)
        if self.hparams.add_distance_loss_to_latent:
            logs, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.hparams.distance_loss_weight
            self.log('{}_dist_loss_btw_latent'.format(self.mode), dist_loss, on_step=False, on_epoch=True)
        if self.hparams.add_consistency_loss:
            logs, cons_loss = self.cons_step(h)
            pretext_loss += cons_loss * self.hparams.consistency_loss_weight
            self.log('{}_cons_loss'.format(self.mode), cons_loss, on_step=False, on_epoch=True)
        if self.hparams.add_MMD_loss:
            logs, MMD_loss = self.MMD_step(h)
            pretext_loss += MMD_loss * self.hparams.MMD_loss_weight
            self.log('{}_MMD_loss'.format(self.mode), MMD_loss, on_step=False, on_epoch=True)
        if self.hparams.add_latent_reconstruction_loss:
            logs, latent_recon_loss = self.latent_recon_step(h)
            pretext_loss += latent_recon_loss * self.hparams.latent_reconstruction_loss_weight
            self.log('{}_latent_recon_loss'.format(self.mode), latent_recon_loss, on_step=False, on_epoch=True)
        if self.hparams.cont_align_loss_criterion != "none":
            if self.hparams.cont_align_loss_criterion in ['barlowtwins', 'clip'] or h[0].shape[0] == self.hparams.batch_size:
                self.cont_pair = 'align'
                logs, cont_loss = self.cont_align_step(h)
                for k, v in logs.items():
                    self.log(k, v, on_step=False, on_epoch=True)
                pretext_loss += self.hparams.cont_align_loss_weight * cont_loss
                if self.hparams.add_cont_type_loss:
                    h_list = self.prepare_cont_type_h(h, batch['y'])
                    for i, h_type in enumerate(h_list):
                        self.cont_pair = 'align_type'
                        logs, cont_loss = self.cont_align_step(h_type)
                        for k, v in logs.items():
                            self.log(k, v, on_step=False, on_epoch=True)
                        pretext_loss += self.hparams.cont_align_loss_weight * cont_loss
            self.log('{}_pretext_loss'.format(self.mode), pretext_loss, on_step=False, on_epoch=True)
        return pretext_loss
    
    def ae_validation_step(self, batch, batch_idx):
        # if self.global_step == 0: 
        #     wandb.define_metric('val_pretext_loss', summary='min')
        #     wandb.define_metric('val_recon_loss', summary='min')
        pretext_loss = 0
        if self.hparams.ae_net == 'ae':
            logs, h, recon_loss = self.ae_step(batch) 
        elif self.hparams.ae_net == 'vae':
            logs, h, recon_loss = self.vae_step(batch)
        pretext_loss += recon_loss
        if self.hparams.add_distance_loss_to_latent:
            _, dist_loss = self.dist_step(h)
            pretext_loss += dist_loss * self.hparams.distance_loss_weight
            logs['{}_dist_loss'.format(self.mode)] = dist_loss
        if self.hparams.add_consistency_loss:
            _, cons_loss = self.cons_step(h)
            pretext_loss += cons_loss * self.hparams.consistency_loss_weight
            logs['{}_cons_loss'.format(self.mode)] = cons_loss
        if self.hparams.add_MMD_loss:
            _, MMD_loss = self.MMD_step(h)
            pretext_loss += MMD_loss * self.hparams.MMD_loss_weight
            logs['{}_MMD_loss'.format(self.mode)] = MMD_loss
        if self.hparams.add_latent_reconstruction_loss:
            _, latent_recon_loss = self.latent_recon_step(h)
            pretext_loss += latent_recon_loss * self.hparams.latent_reconstruction_loss_weight
            logs['{}_latent_recon_loss'.format(self.mode)] = latent_recon_loss
        if self.hparams.cont_align_loss_criterion != "none":
            cont_logs = {}
            if self.hparams.cont_align_loss_criterion in ['barlowtwins', 'clip'] or h[0].shape[0] == self.hparams.batch_size:
                self.cont_pair = 'align'
                cont_pair_logs, cont_loss = self.cont_align_step(h)
                cont_logs.update(cont_pair_logs)
                pretext_loss += self.hparams.cont_align_loss_weight * cont_loss
                if self.hparams.add_cont_type_loss:
                    h_list = self.prepare_cont_type_h(h, batch['y'])
                    for i, h_type in enumerate(h_list):
                        self.cont_pair = 'align_type'
                        cont_pair_logs, cont_loss = self.cont_align_step(h_type)
                        cont_logs.update(cont_pair_logs)
                        pretext_loss += self.hparams.cont_align_loss_weight * cont_loss
            cont_logs['{}_pretext_loss'.format(self.mode)] = pretext_loss
            return {**logs, **cont_logs}
        else:
            return logs

    def ae_validation_epoch_end(self, outputs):
        for key, value in outputs[0].items():
            avg = torch.stack([x[key] for x in outputs if key in x.keys()]).mean()
            self.log(key, avg)
    
    def training_step(self, batch, batch_idx):
        if self.hparams.current_phase == 'p1':
            pretext_loss = self.ae_training_step(batch, batch_idx)
            return pretext_loss
        elif self.hparams.current_phase == 'p2':
            output_dict = self.ds_shared_step(batch)
            return output_dict
        elif self.hparams.current_phase == 'p3':
            pretext_loss = self.ae_training_step(batch, batch_idx)
            output_dict = self.ds_shared_step(batch)
            output_dict['loss'] += self.hparams.cs_pretext_weight * pretext_loss
            return output_dict
    
    def training_epoch_end(self, outputs):
        if self.hparams.current_phase == 'p2' or self.hparams.current_phase == 'p3':
            return self.ds_training_epoch_end(outputs)
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.current_phase == 'p1':
            ae_output_dict = self.ae_validation_step(batch, batch_idx)
            return ae_output_dict
        elif self.hparams.current_phase == 'p2':
            ds_output_dict = self.ds_validation_step(batch, batch_idx)
            return ds_output_dict
        elif self.hparams.current_phase == 'p3':
            ae_output_dict = self.ae_validation_step(batch, batch_idx)
            ds_output_dict = self.ds_validation_step(batch, batch_idx)
            return {
                'ae_output_dict': ae_output_dict,
                'ds_output_dict': ds_output_dict
            }
    
    def validation_epoch_end(self, outputs):
        if self.hparams.current_phase == 'p1':
            self.ae_validation_epoch_end(outputs)
        elif self.hparams.current_phase == 'p2':
            self.ds_validation_epoch_end(outputs)
            if self.hparams.ds_save_latent_val:
                self.save_latent(outputs, 'val')
        elif self.hparams.current_phase == 'p3':
            ae_outputs = [output['ae_output_dict'] for output in outputs]
            ds_outputs = [output['ds_output_dict'] for output in outputs]
            self.ae_validation_epoch_end(ae_outputs)
            self.ds_validation_epoch_end(ds_outputs)
            if self.hparams.ds_save_latent_val:
                self.save_latent(ds_outputs, 'val')

    def ds_shared_step(self, batch):
        if self.hparams.ds_task == 'class':
            output_dict = self.class_step(batch)
            output_dict['loss'] = output_dict['class_loss']
        elif self.hparams.ds_task == 'surv':
            output_dict = self.surv_step(batch)
            output_dict['loss'] = output_dict['surv_loss']
        elif self.hparams.ds_task == 'reg':
            output_dict = self.reg_step(batch)
            output_dict['loss'] = output_dict['reg_loss']
        elif self.hparams.ds_task == 'multi':
            output_dict = self.class_step(batch)
            output_dict.update(self.surv_step(batch))
            output_dict.update(self.reg_step(batch))
            output_dict['loss'] = output_dict['class_loss'] * self.ds_k_class + output_dict['surv_loss'] * self.ds_k_surv + output_dict['reg_loss'] * self.ds_k_reg
        return output_dict

    def class_step(self, batch):
        x_A, x_B, x_C = batch['x']
        y = batch['y']
        sample_ids = batch['sample_id']
        if self.hparams.ds_mask_A:
            x_A = self.ds_mask_x_ch(x_A)
        if self.hparams.ds_mask_B:
            x_B = self.ds_mask_x_ch(x_B)
        h = self.forward((x_A, x_B, x_C))
        if self.hparams.ds_add_omics_identity:
            down_loss = 0
            y_prob_omic = []
            for i, h_omic in enumerate(h):
                omic_id = torch.zeros((h_omic.shape[0], len(h))).to(self.device)
                omic_id[:, i] = 1
                h_omic = torch.cat([h_omic, omic_id], dim=1)
                y_out_omic = self.class_net(h_omic)
                if self.hparams.cl_loss == "wbce":
                    down_loss += self.wbce(y_out_omic, y, self.hparams.class_weights)
                elif self.hparams.cl_loss == "bce":
                    down_loss += self.criterion(y_out_omic, y)
                y_true = y.long()
                y_prob_omic.append(F.softmax(y_out_omic, dim=1))
            y_prob = torch.mean(torch.stack(y_prob_omic), axis=0)
            _, y_pred = torch.max(y_prob, 1)
            h = torch.cat(h, dim=1)
            
        else:
            y_out = self.class_net(h)
            if self.hparams.cl_loss == "wbce":
                down_loss = self.wbce(y_out, y, self.hparams.class_weights)
            elif self.hparams.cl_loss == "bce":
                down_loss = self.criterion(y_out, y)
            y_true = y.long()
            y_prob = F.softmax(y_out, dim=1)
            _, y_pred = torch.max(y_prob, 1)
            
        return {
            "class_loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true": y_true,
            "y_pred": y_pred.detach(),
            "y_prob": y_prob.detach()
        }

    def ds_mask_x_ch(self, x):
        x_masked = []
        for i in range(len(x)):
            if self.hparams.ds_masking_method == 'zero':
                x_masked.append(torch.zeros_like(x[i]))
            elif self.hparams.ds_masking_method == 'gaussian_noise':
                x_masked.append(x[i] + torch.randn_like(x[i]))
        return x_masked
    
    def surv_step(self, batch):
        x_A, x_B, x_C = batch['x']
        sample_ids = batch['sample_id']
        surv_T, surv_E, y_true = batch['survival']
        h = self.forward((x_A, x_B, x_C))
        y_out = self.surv_net(h)
        if self.hparams.survival_loss == 'MTLR':
            down_loss = MTLR_survival_loss(y_out, y_true, surv_E, self.tri_matrix_1)
        predict = self.predict_risk(y_out)
        survival = predict['survival']
        risk = predict['risk']
        return {
            "surv_loss": down_loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "y_true_E": surv_E.detach(),
            "y_true_T": surv_T.detach(),
            "survival": survival.detach(),
            "risk": risk.detach(),
            "y_out": y_out.detach()
        }
    
    def reg_step(self, batch):
        x_A, x_B, x_C = batch['x']
        v = batch['value']
        sample_ids = batch['sample_id']
        h = self.forward((x_A, x_B, x_C))
        v_bar = self.reg_net(h)
        loss = self.reg_loss(v_bar, v)
        return {
            "reg_loss": loss,
            "sample_ids": sample_ids,
            "h": h.detach(),
            "v": v.detach(),
            "v_bar": v_bar.detach()
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
    
    def compute_reg_metrics(self, outputs):
        v = torch.cat([x["v"] for x in outputs]).cpu().numpy()
        v_bar = torch.cat([x["v_bar"] for x in outputs]).cpu().numpy()
        rmse =  sk.metrics.mean_squared_error(v, v_bar, squared=False)
        return rmse
    
    def ds_shared_epoch_end(self, outputs):
        if 'class' in self.ds_tasks:
            class_loss = torch.stack([x["class_loss"] for x in outputs]).mean()
            self.log("{}_class_loss".format(self.mode), class_loss)
            accuracy, precision, recall, f1, auc = self.compute_class_metrics(outputs)
            self.log("{}_accuracy".format(self.mode), accuracy)
            self.log("{}_precision".format(self.mode), precision)
            self.log("{}_recall".format(self.mode), recall)
            self.log("{}_f1".format(self.mode), f1)
            self.log("{}_auc".format(self.mode), auc)
        
        if 'surv' in self.ds_tasks:
            surv_loss = torch.stack([x["surv_loss"] for x in outputs]).mean()
            self.log("{}_surv_loss".format(self.mode), surv_loss)
            c_index, ibs = self.compute_surv_metrics(outputs)
            self.log("{}_c_index".format(self.mode), c_index)
            self.log("{}_ibs".format(self.mode), ibs)
        
        if 'reg' in self.ds_tasks:
            reg_loss = torch.stack([x["reg_loss"] for x in outputs]).mean()
            self.log("{}_reg_loss".format(self.mode), reg_loss)
            rmse = self.compute_reg_metrics(outputs)
            self.log("{}_rmse".format(self.mode), rmse)
        
        if self.hparams.ds_task == 'multi':
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log("{}_down_loss".format(self.mode), avg_loss)
        
        if self.hparams.current_phase == 'p3':
            total_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.log("{}_total_loss".format(self.mode), total_loss)

    def ds_training_epoch_end(self, outputs):
        if self.hparams.ds_save_latent_training:
            self.save_latent(outputs, 'train')
        return self.ds_shared_epoch_end(outputs)
    
    def ds_validation_step(self, batch, batch_idx):
        if self.global_step == 0: 
            if 'class' in self.ds_tasks:
                wandb.define_metric('val_accuracy', summary='max')
                wandb.define_metric('val_precision', summary='max')
                wandb.define_metric('val_recall', summary='max')
                wandb.define_metric('val_f1', summary='max')
                wandb.define_metric('val_auc', summary='max')
        return self.ds_shared_step(batch)

    def ds_validation_epoch_end(self, outputs):
        return self.ds_shared_epoch_end(outputs)
    
    def test_step(self, batch, batch_idx):
        if self.global_step == 0: 
            wandb.define_metric('test_accuracy', summary='max')
        return self.ds_shared_step(batch)
    
    def test_epoch_end(self, outputs):
        self.ds_shared_epoch_end(outputs)
        if self.hparams.ds_save_latent_testing:
            self.save_latent(outputs, 'test')
    
    def save_latent(self, outputs, mode):
        sample_ids_list = []
        for x in outputs:
            sample_ids_list.extend(x["sample_ids"])
        h_concat = torch.cat([x["h"] for x in outputs]).cpu().numpy()
        latent_space = pd.DataFrame(h_concat, index=sample_ids_list)
        # latent_space.to_csv(os.path.join(self.checkpoint_path, 'latent_space.tsv'), sep='\t')
        latent_space.to_csv('{}_latent_space.tsv'.format(mode), sep='\t')
    
    def predict_step(self, batch, batch_idx):
        return self.ds_shared_step(batch)

    def on_predict_epoch_end(self, results):
        outputs = results[0]
        if self.hparams.ds_task == 'class':
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
        
        elif self.hparams.ds_task == "surv":
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
            ones_matrix = torch.ones(self.hparams.time_num, self.hparams.time_num + 1, device=self.device)
        else:
            ones_matrix = torch.ones(self.hparams.time_num + 1, self.hparams.time_num + 1, device=self.device)
        tri_matrix = torch.tril(ones_matrix).cuda()
        return tri_matrix
    
    def predict_risk(self, y_out):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        if self.hparams.survival_loss == 'MTLR':
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
        time_points = np.linspace(0, T_max * (1 + extra_time_percent), self.hparams.time_num + 1)

        return time_points
        