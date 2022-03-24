from matplotlib.pyplot import cla
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch

class ABCDataset(Dataset):
    def __init__(self, param, A_df, B_df, C_df, labels, indices):
        super().__init__()
        self.param = param
        self.A_tensors = torch.tensor(A_df.values).float()
        if param.split_B:
            self.B_tensors = [torch.tensor(B_ch.values).float() for B_ch in B_df]
        else:
            self.B_tensors = torch.tensor(B_df.values).float()
        self.C_tensors = torch.tensor(C_df.values).float()
        self.label_tensors = torch.tensor(labels.values).float()

        # self.A_df = A_df
        # self.B_df = B_df
        # self.C_df = C_df
        # self.labels = labels
        # self.indices = indices
    
    def __len__(self):
        return len(self.label_tensors)
    
    def __getitem__(self, idx):
        # A_sample = torch.tensor(self.A_df.iloc[:, self.indices[idx]]).float()
        # if self.param.split_B:
        #     B_sample = [torch.tensor(B_ch.iloc[:, self.indices[idx]]).float() for B_ch in self.B_df]
        # else:
        #     B_sample = torch.tensor(self.B_df.iloc[:, self.indices[idx]]).float()
        # C_sample = torch.tensor(self.C_df.iloc[:, self.indices[idx]]).float()
        # label = torch.tensor(self.labels.iloc[self.indices[idx], :]).float()

        A_sample = self.A_tensors[:, idx]
        if self.param.split_B:
            B_sample = [self.B_tensors[ch][:, idx] for ch in range(len(self.B_tensors))]
        else:
            B_sample = self.B_tensors[:, idx]
        C_sample = self.C_tensors[:, idx]
        label = self.label_tensors[idx,:]
        return A_sample, B_sample, C_sample, label
        
