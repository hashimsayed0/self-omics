from posixpath import split
from matplotlib.pyplot import cla
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch

class ABCDataset(Dataset):
    def __init__(self, A_df, B_df, C_df, indices, split_A, split_B, ds_tasks, labels = None, survival_T_array = None, survival_E_array = None, y_true_tensor = None, values = None):
        super().__init__()
        self.split_A = split_A
        self.split_B = split_B
        self.indices = indices
        self.ds_tasks = ds_tasks

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

        if 'class' in self.ds_tasks:
            labels = labels.iloc[indices]
            self.label_tensors = torch.tensor(labels.values.astype(float)).float()
            self.sample_ids = labels.index.to_list()
        
        if 'surv' in self.ds_tasks:
            self.survival_T_array = survival_T_array[indices]
            self.survival_E_array = survival_E_array[indices]
            self.y_true_tensor = y_true_tensor[indices]
        
        if 'reg' in self.ds_tasks:
            values = values.iloc[indices]
            self.value_tenors = torch.tensor(values.values.astype(float)).float()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_dict = {}
        if self.split_A:
            A_sample = [self.A_tensors[ch][:, idx] for ch in range(len(self.A_tensors))]
        else:
            A_sample = self.A_tensors[:, idx]
        if self.split_B:
            B_sample = [self.B_tensors[ch][:, idx] for ch in range(len(self.B_tensors))]
        else:
            B_sample = self.B_tensors[:, idx]
        C_sample = self.C_tensors[:, idx]

        data_dict['x'] = (A_sample, B_sample, C_sample)
        data_dict['sample_id'] = self.sample_ids[idx]
        
        if 'class' in self.ds_tasks:
            label = self.label_tensors[idx,:]
            data_dict['y'] = label

        if 'surv' in self.ds_tasks:
            survival_T = self.survival_T_array[idx]
            survival_E = self.survival_E_array[idx]
            y_true = self.y_true_tensor[idx]
            data_dict['survival'] = (survival_T, survival_E, y_true)
        
        if 'reg' in self.ds_tasks:
            value = self.value_tenors[idx]
            data_dict['value'] = value

        return data_dict        
        
