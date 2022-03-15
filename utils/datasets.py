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
        self.A_df = A_df
        self.B_df = B_df
        self.C_df = C_df
        self.labels = labels
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        A_sample = torch.tensor(self.A_df.iloc[:, self.indices[idx]]).float()
        if self.param.split_B:
            B_sample = [torch.tensor(B_ch.iloc[:, self.indices[idx]]).float() for B_ch in self.B_df]
        else:
            B_sample = torch.tensor(self.B_df.iloc[:, self.indices[idx]]).float()
        C_sample = torch.tensor(self.C_df.iloc[:, self.indices[idx]]).float()
        label = torch.tensor(self.labels.iloc[self.indices[idx]]).float()
        return A_sample, B_sample, C_sample, label
        
