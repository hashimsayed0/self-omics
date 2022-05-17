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
        return loss, on_diag, self.lambd * off_diag, z1, z2


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class NTXentLoss(torch.nn.Module):
    """
    Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
    """

    def __init__(self, batch_size, temperature, similarity, latent_dim, normalize, p_norm):
        super(NTXentLoss, self).__init__()
        # Batch size
        self.batch_size = batch_size
        # Temperature to use scale logits
        self.temperature = temperature
        # initialize softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if similarity == 'cosine' else self._dot_simililarity
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        # Two-Layer Projection Network
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(latent_dim, latent_dim)
        # Last linear layer for final projection
        self.linear_layer2 = nn.Linear(latent_dim, latent_dim)
        
        self.normalize = normalize
        self.p_norm = p_norm
        

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        # mask = torch.from_numpy((diagonal + q1 + q3))
        self.register_buffer('mask', torch.from_numpy((diagonal + q1 + q3)))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        self.mask = (1 - self.mask).type(torch.bool)
        # Transfer the mask to the device and return
        return self.mask #.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = torch.nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)

    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)
        # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
        l_pos = torch.diag(similarity, self.batch_size)
        # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
        r_pos = torch.diag(similarity, -self.batch_size)
        # Concatenate all positive samples as a 2nx1 column vector
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
        negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
        # Concatenate positive samples as the first column to negative samples array
        logits = torch.cat((positives, negatives), dim=1)
        # Normalize logits via temperature
        logits /= self.temperature
        # Labels are all zeros since all positive samples are the 0th column in logits array.
        # So we will select positive samples as numerator in NTXentLoss
        labels = torch.zeros(2 * self.batch_size).to(logits.device).long() 
        # self.register_buffer('labels', torch.zeros(2 * self.batch_size).long())
        # Compute total loss
        loss = self.criterion(logits, labels)
        # Loss per sample
        closs = loss / (2 * self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, h1, h2):
        """

        Args:
            representation (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):

        """
        representation = torch.cat((h1, h2), dim=0)
        # Forward pass on Projection
        # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        z = F.leaky_relu(self.linear_layer1(representation))
        # Apply final linear layer
        z = self.linear_layer2(z)
        # Do L2 normalization
        z = F.normalize(z, p=self.p_norm, dim=1) if self.normalize else z
        closs = self.XNegloss(z)
        z1, z2 = torch.split(representation, self.batch_size)
        return closs, z1, z2


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
