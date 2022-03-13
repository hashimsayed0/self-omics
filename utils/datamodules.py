from pytorch_lightning import LightningDataModule
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .datasets import ABCDataset
from torch.utils.data import DataLoader

class ABCDataModule(LightningDataModule):
    def __init__(self, param, current_fold):
        super().__init__()
        self.param = param
        self.current_fold = current_fold
        self.load_data()
    
    def load_data(self):
        self.A_df = self.load_file('A')
        self.B_df = self.load_file('B')
        self.C_df = self.load_file('C')

        if self.param.use_sample_list:
            sample_list_path = os.path.join(self.param.data_dir, 'sample_list.tsv')
            print('Loading sample list from ' + sample_list_path)
            self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        else:
            sample_list = self.A_df.columns

        self.A_df = self.A_df.loc[:, sample_list]
        self.B_df = self.B_df.loc[:, sample_list]
        self.C_df = self.C_df.loc[:, sample_list]

        labels_path = os.path.join(self.param.data_dir, 'labels.tsv')
        self.labels = pd.read_csv(labels_path, sep='\t', header=0, index_col=0)
        self.labels = self.labels.loc[:, sample_list]

        kf = StratifiedKFold(n_splits=self.param.num_folds, random_state=self.param.seed)
        for i, (train_index, test_index) in enumerate(kf.split(self.A_df, self.labels)):
            if i == self.current_fold:
                self.train_index = train_index[:int(self.param.train_val_split * len(train_index))]
                self.val_index = train_index[int(self.param.train_val_split * len(train_index)):]
                self.test_index = test_index
                break
        
    def load_file(self, file_name):
        file_path = os.path.join(self.param.data_dir, file_name + '.npy')
        print('Loading data from ' + file_path)
        values = np.load(file_path, allow_pickle=True)
        features_path = os.path.join(self.param.data_dir, file_name + '_features.npy')
        print('Loading features from ' + features_path)
        features = np.load(features_path, allow_pickle=True)
        samples_path = os.path.join(self.param.data_dir, file_name + '_samples.npy')
        print('Loading samples from ' + samples_path)
        samples = np.load(samples_path, allow_pickle=True)
        df = pd.DataFrame(data=values, index=features, columns=samples)
        return df

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.trainset = ABCDataset(self.param, self.A_df, self.B_df, self.C_df, self.labels, self.train_index)
            self.valset = ABCDataset(self.param, self.A_df, self.B_df, self.C_df, self.labels, self.val_index)
        
        if stage == "test" or stage is None:
            self.testset = ABCDataset(self.param, self.A_df, self.B_df, self.C_df, self.labels, self.test_index)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.param.batch_size, num_workers=self.param.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.param.batch_size, num_workers=self.param.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.param.batch_size, num_workers=self.param.num_workers)
