from posixpath import split
from matplotlib.pyplot import cla
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch

class ABCDataset(Dataset):
    def __init__(self, A_df, B_df, C_df, labels, indices, split_A, split_B):
        super().__init__()
        self.split_A = split_A
        self.split_B = split_B

        if self.split_A:
            A_df = [A_df[ch].iloc[:, indices] for ch in range(len(A_df))]
            self.A_tensors = [torch.tensor(A_ch.values.astype(float)).float() for A_ch in A_df]
        else:
            self.A_tensors = torch.tensor(A_df.values.astype(float)).float()
        
        if self.split_B:
            B_df = [B_df[ch].iloc[:, indices] for ch in range(len(B_df))]
            self.B_tensors = [torch.tensor(B_ch.values.astype(float)).float() for B_ch in B_df]
        else:
            self.B_tensors = torch.tensor(B_df.values.astype(float)).float()
        
        C_df = C_df.iloc[:, indices]
        self.C_tensors = torch.tensor(C_df.values.astype(float)).float()
        labels = labels.iloc[indices]
        self.label_tensors = torch.tensor(labels.values.astype(float)).float()
    
    def __len__(self):
        return len(self.label_tensors)
    
    def __getitem__(self, idx):
        if self.split_A:
            A_sample = [self.A_tensors[ch][:, idx] for ch in range(len(self.A_tensors))]
        else:
            A_sample = self.A_tensors[:, idx]
        if self.split_B:
            B_sample = [self.B_tensors[ch][:, idx] for ch in range(len(self.B_tensors))]
        else:
            B_sample = self.B_tensors[:, idx]
        C_sample = self.C_tensors[:, idx]
        label = self.label_tensors[idx,:]
        return A_sample, B_sample, C_sample, label
        
