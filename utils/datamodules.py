from email.policy import default
from aiohttp import worker
from pytorch_lightning import LightningDataModule
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from .datasets import ABCDataset
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
from .preprocessing import select_features

class ABCDataModule(LightningDataModule):
    def __init__(self, current_fold, num_folds, seed, data_dir, use_sample_list, batch_size, val_ratio, num_workers, split_A, split_B, feature_selection, feature_selection_alpha, feature_selection_percentile, **config):
        super().__init__()
        self.current_fold = current_fold
        self.num_folds = num_folds
        self.data_dir = data_dir
        self.use_sample_list = use_sample_list
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.split_A = split_A
        self.split_B = split_B
        self.seed = seed
        self.feature_selection = feature_selection
        self.feature_selection_alpha = feature_selection_alpha
        self.feature_selection_percentile = feature_selection_percentile
        # self.save_hyperparameters()
        self.load_data()
        self.preprocess_data()

    @staticmethod
    def add_data_module_args(parent_parser):
        parser = parent_parser.add_argument_group("ABCDataModule")
        parser.add_argument("--data_dir", type=str, default="./data/ABC_inter/Normalized",
                            help="directory containing the dataset")
        parser.add_argument('--use_sample_list', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='provide a subset sample list of the dataset, store in the path data_dir/sample_list.tsv, if False use all the samples')
        parser.add_argument('--batch_size', type=int, default=32,
                                help='data batch size')
        parser.add_argument('--val_ratio', type=float, default=0.15,
                                help='val proportion of total training data')
        parser.add_argument('--num_workers', type=int, default=0,
                                help='number of workers for data loading')
        parser.add_argument('--split_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, A is split into 23 parts corresponding to the 23 different chromosomes')                        
        parser.add_argument('--split_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, B is split into 23 parts corresponding to the 23 different chromosomes')
        

        parser.add_argument("--feature_selection", type=str, default="none", help="options: none, f_test, chi2, mutual_info, all")
        parser.add_argument("--feature_selection_alpha", type=float, default=0.01)
        parser.add_argument("--feature_selection_percentile", type=float, default=10)
        return parent_parser

    def load_data(self):
        self.A_df = self.load_file('A')
        self.B_df = self.load_file('B')
        self.C_df = self.load_file('C')

        if self.use_sample_list:
            sample_list_path = os.path.join(self.data_dir, 'sample_list.tsv')
            print('Loading sample list from ' + sample_list_path)
            sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        else:
            sample_list = self.A_df.columns

        self.A_df = self.A_df.loc[:, sample_list]
        if self.split_A:
            self.A_df, _ = self.separate_A(self.A_df)
        self.B_df = self.B_df.loc[:, sample_list]
        if self.split_B:
            self.B_df, _ = self.separate_B(self.B_df)
        self.C_df = self.C_df.loc[:, sample_list]

        labels_path = os.path.join(self.data_dir, 'labels.tsv')
        self.labels = pd.read_csv(labels_path, sep='\t', header=0, index_col=0)
        self.labels = self.labels.loc[sample_list]

    def preprocess_data(self):
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(kf.split(self.C_df.T, self.labels)):
            if i == self.current_fold:
                self.train_index, self.val_index = train_test_split(train_index, test_size=self.val_ratio, random_state=self.seed, stratify=self.labels.iloc[train_index])
                self.test_index = test_index
                break

        if self.feature_selection != "none":
            self.A_df = select_features(self.A_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile)
            self.C_df = select_features(self.C_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile)
            if self.split_B:
                self.B_df_list = []
                for B_df in self.B_df:
                    self.B_df_list.append(select_features(B_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile))
                self.B_df = self.B_df_list

        self.class_weights = self.calculate_class_weights(self.labels.iloc[self.train_index].values)
        self.labels = pd.get_dummies(self.labels.iloc[:,0])
        
    def load_file(self, file_name):
        file_path = os.path.join(self.data_dir, file_name + '.npy')
        print('Loading data from ' + file_path)
        values = np.load(file_path, allow_pickle=True)
        features_path = os.path.join(self.data_dir, file_name + '_features.npy')
        print('Loading features from ' + features_path)
        features = np.load(features_path, allow_pickle=True)
        samples_path = os.path.join(self.data_dir, file_name + '_samples.npy')
        print('Loading samples from ' + samples_path)
        samples = np.load(samples_path, allow_pickle=True)
        df = pd.DataFrame(data=values, index=features, columns=samples)
        return df

    def separate_B(self, B_df_single):
        """
        Separate the DNA methylation dataframe into subsets according to their targeting chromosomes

        Parameters:
            B_df_single(DataFrame) -- a dataframe that contains the single DNA methylation matrix

        Return:
            B_df_list(list) -- a list with 23 subset dataframe
            B_dim(list) -- the dims of each chromosome
        """
        anno = pd.read_csv('./anno/B_anno.csv', dtype={'CHR': str}, index_col=0)
        anno_contain = anno.loc[B_df_single.index, :]
        print('Separating B according to the targeting chromosome...')
        B_df_list, B_dim_list = [], []
        ch_id = list(range(1, 23))
        ch_id.append('X')
        for ch in ch_id:
            ch_index = anno_contain[anno_contain.CHR == str(ch)].index
            ch_df = B_df_single.loc[ch_index, :]
            B_df_list.append(ch_df)
            B_dim_list.append(len(ch_df))

        return B_df_list, B_dim_list
    
    def separate_A(self, A_df):
        """
        Separate the RNA-seq dataframe into subsets according to their targeting chromosomes

        Parameters:
            A_df(DataFrame) -- a dataframe that contains the single RNA-seq matrix

        Return:
            A_df_list(list) -- a list with 23 subset dataframe
            A_dim(list) -- the dims of each chromosome
        """
        anno = pd.read_csv('./anno/A_anno2.txt', dtype={'CHR': str}, index_col=0, sep='\t')
        genes = sorted([g for g in A_df.index if g.split('.')[0] in anno.index])
        anno = anno.sort_index()
        anno.index = genes
        anno_contain = anno.loc[:, :]
        print('Separating A according to the targeting chromosome...')
        A_df_list, A_dim_list = [], []
        ch_id = list(range(1, 23))
        ch_id.append('X')
        for ch in ch_id:
            ch_index = anno_contain[anno_contain.CHR == str(ch)].index
            ch_df = A_df.loc[ch_index, :]
            A_df_list.append(ch_df)
            A_dim_list.append(len(ch_df))

        return A_df_list, A_dim_list

    def calculate_class_weights(self, y_train):
        y_train = y_train.astype(int)[:,0]
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        # class_weights = class_weights / np.sum(class_weights)
        return class_weights

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.trainset = ABCDataset(self.A_df, self.B_df, self.C_df, self.labels, self.train_index, self.split_A, self.split_B)
            self.valset = ABCDataset(self.A_df, self.B_df, self.C_df, self.labels, self.val_index, self.split_A, self.split_B)
        
        if stage == "test" or stage is None:
            self.testset = ABCDataset(self.A_df, self.B_df, self.C_df, self.labels, self.test_index, self.split_A, self.split_B)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers)
