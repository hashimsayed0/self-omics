from matplotlib import axis
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

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