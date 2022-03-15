from pyrsistent import v
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import Encoder, Decoder, ProjectionHead, AENet
from .losses import SimCLR_Loss
import wandb


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size_A, input_size_B, input_size_C, latent_size, lr, weight_decay, momentum, drop_p, loss_temperature, split_B, batch_size, **config):
        super(AutoEncoder, self).__init__()
        self.input_size_A = input_size_A
        self.input_size_B = input_size_B
        self.input_size_C = input_size_C
        self.latent_size = latent_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.drop_p = drop_p
        self.aenet = AENet((input_size_A, input_size_B, input_size_C), latent_size, split_B, dropout_p=drop_p)
        self.projector = ProjectionHead(latent_size, latent_size, latent_size // 2)
        self.cont_criterion = SimCLR_Loss(batch_size = batch_size, temperature = loss_temperature)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument("--latent_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--drop_p", type=float, default=0.2)
        parser.add_argument("--loss_temperature", type=float, default=0.5)
        return parent_parser

    def forward(self, x):
        return self.enc_A(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def training_step(self, batch, batch_idx):
        x_A, x_B, x_C, _ = batch
        h_A, h_B, h_C = self.aenet.encode((x_A, x_B, x_C))
        x_A_recon, x_B_recon, x_C_recon = self.aenet.decode((h_A, h_B, h_C))
        recon_loss = F.mse_loss(x_A_recon, x_A) + F.mse_loss(x_B_recon, x_B) + F.mse_loss(x_C_recon, x_C)
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