from matplotlib import axis
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .networks import SimCLRProjectionHead


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature, latent_size, proj_size):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.projector = SimCLRProjectionHead(latent_size, latent_size, proj_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, h_i, h_j):
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

# class CLIPLoss(nn.Module):
#     def __init__(
#         self,
#         temperature
#     ):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z_i, z_j):

#         # Calculating the Loss
#         logits = (z_j @ z_i.T) / self.temperature
#         z_i_similarity = z_i @ z_i.T
#         z_j_similarity = z_j @ z_j.T
#         targets = F.softmax(
#             (z_i_similarity + z_j_similarity) / 2 * self.temperature, dim=-1
#         )
#         z_j_loss = cross_entropy(logits, targets, reduction='none')
#         z_i_loss = cross_entropy(logits.T, targets.T, reduction='none')
#         loss =  (z_i_loss + z_j_loss) / 2.0 # shape: (batch_size)
#         return loss.mean()


class CLIPLoss(nn.Module):
    def __init__(self, temperature, latent_size, proj_size):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.W_i = nn.Parameter(torch.randn(latent_size, proj_size))
        self.W_j = nn.Parameter(torch.randn(latent_size, proj_size))
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, z_i, z_j):
        z_i_proj = z_i @ self.W_i
        z_i_proj_norm = z_i_proj / torch.norm(z_i_proj, dim=1).unsqueeze(1)
        z_j_proj = z_j @ self.W_j
        z_j_proj_norm = z_j_proj / torch.norm(z_j_proj, dim=1).unsqueeze(1)

        logits = (z_i_proj_norm @ z_j_proj_norm.T) * torch.exp(self.temperature)

        labels = torch.arange(logits.shape[0]).to(logits.device).long()
        loss_i = self.ce(logits, labels)
        loss_j = self.ce(logits.T, labels)
        loss = (loss_i + loss_j) / 2.0

        return loss.mean(), loss_i.mean(), loss_j.mean()
        

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd, latent_size, proj_size):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(proj_size, affine=False)
        self.projector = SimCLRProjectionHead(latent_size, latent_size // 2, proj_size)

    def forward(self, y1, y2):
        n = y1.shape[0]
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2) / n

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss, on_diag, self.lambd * off_diag


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def weighted_binary_cross_entropy(input, target, weights=None):

    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
    return torch.mean(bce)

def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

def MTLR_survival_loss(y_pred, y_true, E, tri_matrix, reduction='mean'):
    """
    Compute the MTLR survival loss
    """
    # Get censored index and uncensored index
    censor_idx = []
    uncensor_idx = []
    for i in range(len(E)):
        # If this is a uncensored data point
        if E[i] == 1:
            # Add to uncensored index list
            uncensor_idx.append(i)
        else:
            # Add to censored index list
            censor_idx.append(i)

    # Separate y_true and y_pred
    y_pred_censor = y_pred[censor_idx]
    y_true_censor = y_true[censor_idx]
    y_pred_uncensor = y_pred[uncensor_idx]
    y_true_uncensor = y_true[uncensor_idx]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = - (torch.sum(torch.log(reduc_phi_censor)) + torch.sum(torch.log(reduc_phi_uncensor)) - torch.sum(torch.log(reduc_z_censor)) - torch.sum(torch.log(reduc_z_uncensor)))

    if reduction == 'mean':
        loss = loss / E.shape[0]

    return loss
