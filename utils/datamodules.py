from email.policy import default
from pytorch_lightning import LightningDataModule
import os
import numpy as np
import pandas as pd
from .preprocessing import select_features, scale_features
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .datasets import ABCDataset
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
import torch
from wandb import wandb

class OmicsDataModule(LightningDataModule):
    def __init__(self, current_fold, num_folds, seed, data_dir, use_sample_list, batch_size, val_ratio, num_workers, split_A, split_B, feature_selection, feature_selection_alpha, feature_selection_percentile, **config):
        super().__init__()
        self.current_fold = current_fold
        self.num_folds = num_folds
        self.data_dir = data_dir
        self.use_sample_list = use_sample_list
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.data_format = config['data_format']
        self.split_A = split_A
        self.split_B = split_B
        self.augment_B = config['augment_B']
        self.augment_A = config['augment_A']
        self.seed = seed
        self.data_scaler = config['data_scaler']
        self.feature_selection = feature_selection
        self.feature_selection_alpha = feature_selection_alpha
        self.feature_selection_percentile = feature_selection_percentile
        self.ds_task = config['ds_task']
        self.survival_loss = config['survival_loss']
        self.survival_T_max = config['survival_T_max']
        self.time_num = config['time_num']
        self.ds_task = config['ds_task']
        if self.ds_task == 'multi':
            self.ds_tasks = ['class', 'surv', 'reg']
        else:
            self.ds_tasks = [self.ds_task]
        self.use_test_as_val_for_downstream = config['use_test_as_val_for_downstream']
        self.prediction_data = config['prediction_data']
        self.mode = 'pretraining'
        self.train_in_phases = config['train_in_phases']
        config['p3_data_ratio'] = config['p2_data_ratio'] ### IMPORTANT: Remove after debugging
        self.phase = 'p1'
        self.phases_data_ratios = {
            'p1': config['p1_data_ratio'],
            'p2': config['p2_data_ratio'],
            'p3': config['p3_data_ratio'],
        }
        self.train_downstream_on_some_types = config['train_downstream_on_some_types']
        self.downstream_cancer_types = config['downstream_cancer_types']
        self.pretraining_data_ratio = config['pretraining_data_ratio']
        self.downstream_data_ratio = config['downstream_data_ratio']
        self.omics_types = config['omics_types']
        if self.omics_types == 'abc':
            self.use_a, self.use_b, self.use_c = True, True, True
        elif self.omics_types == 'a':
            self.use_a, self.use_b, self.use_c = True, False, False
            self.B_df, self.C_df = None, None
        elif self.omics_types == 'b':
            self.use_a, self.use_b, self.use_c = False, True, False
            self.A_df, self.C_df = None, None
        elif self.omics_types == 'c':
            self.use_a, self.use_b, self.use_c = False, False, True
            self.A_df, self.B_df = None, None
        # self.save_hyperparameters()
        self.load_data()
        self.preprocess_data()

    @staticmethod
    def add_data_module_args(parent_parser):
        parser = parent_parser.add_argument_group("ABCDataModule")
        parser.add_argument("--data_dir", type=str, default="./data",
                            help="directory containing the dataset")
        parser.add_argument('--data_format', type=str, default='npy',
                            help='the format in which data is stored, options: [npy, tsv]')
        parser.add_argument('--omics_types', type=str, default='abc',
                            help='the omics types to be used, options: [abc, a, b, c], where "a" is gene expression, "b" is DNA methylation and "c" is miRNA expression')
        parser.add_argument('--use_sample_list', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='provide a subset sample list of the dataset, store in the path data_dir/sample_list.tsv, if False use all the samples')
        parser.add_argument('--batch_size', type=int, default=512,
                                help='data batch size')
        parser.add_argument('--val_ratio', type=float, default=0.15,
                                help='val proportion of total training data')
        parser.add_argument('--use_test_as_val_for_downstream', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='use test data as val data for the downstream task')
        parser.add_argument('--num_workers', type=int, default=0,
                                help='number of workers for data loading')
        parser.add_argument('--split_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, A is split into 23 parts corresponding to the 23 different chromosomes')                        
        parser.add_argument('--split_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, B is split into 23 parts corresponding to the 23 different chromosomes')
        parser.add_argument('--augment_B', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, B is added with zeros for samples that have data for A and C, but not B, should only be used with ds_mask_B')
        parser.add_argument('--augment_A', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, A is added with zeros for samples that have data for B and C, but not A, should only be used with ds_mask_A')
        parser.add_argument('--train_in_phases', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, train in 3 phases, otherwise train in one epoch')
        parser.add_argument('--p1_data_ratio', type=float, default=1.0,
                                help='ratio of training data to be used for phase 1')
        parser.add_argument('--p2_data_ratio', type=float, default=1.0,
                                help='ratio of training data to be used for phase 2')
        parser.add_argument('--p3_data_ratio', type=float, default=1.0,
                                help='ratio of training data to be used for phase 3')
        parser.add_argument('--train_downstream_on_some_types', default=False, type=lambda x: (str(x).lower() == 'true'),
                                help='if True, train downstream network on some cancer types, otherwise train on all cancer types or a proportion of it based on the param "downstream_data_ratio"')
        parser.add_argument('--downstream_cancer_types', type=str, default='all',
                                help='cancer types to train downstream on, if "train_downstream_on_some_types" is set to True; options: ["all", "n_least_common" (n least common cancer types where 1 <= n <= 33), "custom" (should be given in a file named "downstream_cancer_types.tsv" in the data directory)]')
        parser.add_argument('--pretraining_data_ratio', type=float, default=1.0,
                                help='ratio of training data to be used for pretraining')
        parser.add_argument('--downstream_data_ratio', type=float, default=1.0,
                                help='ratio of training data to be used for downstream')
        parser.add_argument("--feature_selection", type=str, default="none", help="options: none, f_test, chi2, mutual_info, all")
        parser.add_argument("--feature_selection_alpha", type=float, default=0.01)
        parser.add_argument("--feature_selection_percentile", type=float, default=10)
        parser.add_argument("--data_scaler", type=str, default="none", help="options: standard, minmax, none")
        return parent_parser

    def load_data(self):
        if self.use_a:
            self.A_df = self.load_file('A')
            self.sample_list = self.A_df.columns
        if self.use_b:
            self.B_df = self.load_file('B')
            self.sample_list = self.B_df.columns
        if self.use_c:
            self.C_df = self.load_file('C')
            self.sample_list = self.C_df.columns

        if self.use_sample_list:
            # if self.augment_B:
            #     sample_list_folder = 'AC_inter'
            # elif self.augment_A:
            #     sample_list_folder = 'BC_inter'
            # else:
            #     sample_list_folder = 'ABC_inter'
            # sample_list_path = os.path.join(self.data_dir, 'sample_lists', self.ds_task, sample_list_folder, 'self.sample_list.tsv')
            sample_list_path = os.path.join(self.data_dir, 'sample_list.tsv')
            print('Loading sample list from ' + sample_list_path)
            self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        
        if self.use_a:
            self.A_df = self.A_df.loc[:, self.sample_list]
            if self.augment_A:
                print('Augmenting A with zeros')
                samp_min_A = list(set(self.sample_list).difference(set(self.A_df.columns.to_list())))
                aug_df = pd.DataFrame(np.zeros((self.A_df.shape[0],len(samp_min_A))), columns=samp_min_A, index=self.A_df.index)
                self.A_df = pd.concat([self.A_df, aug_df], axis=1)
        
        if self.use_b:
            self.B_df = self.B_df.loc[:, self.sample_list]
            if self.augment_B:
                print('Augmenting B with zeros')
                samp_min_B = list(set(self.sample_list).difference(set(self.B_df.columns.to_list())))
                aug_df = pd.DataFrame(np.zeros((self.B_df.shape[0],len(samp_min_B))), columns=samp_min_B, index=self.B_df.index)
                self.B_df = pd.concat([self.B_df, aug_df], axis=1)
            
        if self.use_c:
            self.C_df = self.C_df.loc[:, self.sample_list]

        self.survival_T_array = None
        self.survival_E_array = None
        self.y_true_tensor = None
        self.values = None
        self.labels = None

        if 'class' in self.ds_tasks:
            labels_path = os.path.join(self.data_dir, 'labels.tsv')
            self.labels = pd.read_csv(labels_path, sep='\t', header=0, index_col=0)
            self.labels = self.labels.loc[self.sample_list]

            tumour_index_path = os.path.join(self.data_dir, 'tumour_index.csv')
            self.tumour_index = pd.read_csv(tumour_index_path, index_col=0)
            self.classes = self.tumour_index['Tumour type']

        if 'surv' in self.ds_tasks:
            survival_path = os.path.join(self.data_dir, 'survival.tsv')   # get the path of the survival data
            survival_df = pd.read_csv(survival_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.survival_T_array = survival_df.iloc[:, -2].astype(float).values
            self.survival_E_array = survival_df.iloc[:, -1].values
            self.survival_T_max = self.survival_T_array.max()
            self.survival_T_min = self.survival_T_array.min()
            if self.survival_loss == 'MTLR':
                self.y_true_tensor = self.get_survival_y_true(self.survival_T_array, self.survival_E_array)
        
        if 'reg' in self.ds_tasks:
            values_path = os.path.join(self.data_dir, 'values.tsv')
            self.values = pd.read_csv(values_path, sep='\t', header=0, index_col=0)
            self.values = self.values.loc[self.sample_list]
        
        if self.train_downstream_on_some_types:
            if self.downstream_cancer_types == 'custom':
                cancer_types_path = os.path.join(self.data_dir, 'downstream_cancer_types.tsv')
                print('Loading downstream cancer types from ' + cancer_types_path)
                ds_cancer_types = np.loadtxt(cancer_types_path, delimiter='\t', dtype='<U32')
                ds_tumour_indices = self.tumour_index['Index'][self.tumour_index['Tumour type'].isin(ds_cancer_types)]
            elif self.downstream_cancer_types == 'all':
                ds_tumour_indices = np.arange(len(self.classes))
            elif self.downstream_cancer_types.endswith('least_common'):
                n = int(self.downstream_cancer_types.split('_')[0])
                ds_tumour_indices = self.labels['sample_type.samples'].value_counts()[-n:].index.to_list()
            self.all_indices = np.arange(self.labels.shape[0])[self.labels.isin(ds_tumour_indices)['sample_type.samples']]

    def preprocess_data(self):
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(kf.split(self.sample_list, self.labels)):
            if i == self.current_fold:
                self.train_index, self.val_index = train_test_split(train_index, test_size=self.val_ratio, random_state=self.seed, stratify=self.labels.iloc[train_index])
                self.test_index = test_index
                break

        if self.data_scaler != 'none':
            if self.data_scaler == 'standard':
                scaler = StandardScaler()
            elif self.data_scaler == 'minmax':
                scaler = MinMaxScaler()
            if self.use_a:
                print('Scaling A')
                self.A_df = scale_features(scaler, self.A_df, self.train_index, self.val_index, self.test_index)
            if self.use_b:
                print('Scaling B')
                self.B_df = scale_features(scaler, self.B_df, self.train_index, self.val_index, self.test_index)
            if self.use_c:
                print('Scaling C')
                self.C_df = scale_features(scaler, self.C_df, self.train_index, self.val_index, self.test_index)

        if self.use_a and self.split_A:
                self.A_df, _ = self.separate_A(self.A_df)
        if self.use_b and self.split_B:
                self.B_df, _ = self.separate_B(self.B_df)

        if self.feature_selection != "none":
            if self.use_a:
                self.A_df = select_features(self.A_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile)
            if self.use_c:
                self.C_df = select_features(self.C_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile)
            if self.use_b and self.split_B:
                    self.B_df_list = []
                    for B_df in self.B_df:
                        self.B_df_list.append(select_features(B_df, self.labels, self.train_index, self.val_index, self.test_index, self.feature_selection, self.feature_selection_alpha, self.feature_selection_percentile))
                    self.B_df = self.B_df_list

        self.class_weights = self.calculate_class_weights(self.labels.iloc[self.train_index].values)
        self.labels = pd.get_dummies(self.labels.iloc[:,0])
        
    def load_file(self, file_name):
        if self.data_format == 'npy':
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
        elif self.data_format == 'tsv':
            file_path = os.path.join(self.data_dir, file_name + '.tsv')
            print('Loading data from ' + file_path)
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=0)
        return df
    

    def get_survival_y_true(self, T, E):
        """
        Get y_true for survival prediction based on T and E
        """
        # Get T_max
        if self.survival_T_max == -1:
            T_max = T.max()
        else:
            T_max = self.survival_T_max

        # Get time points
        time_points = self.get_time_points(T_max)

        # Get the y_true
        y_true = []
        for i, (t, e) in enumerate(zip(T, E)):
            y_true_i = np.zeros(self.time_num + 1)
            dist_to_time_points = [abs(t - point) for point in time_points[:-1]]
            time_index = np.argmin(dist_to_time_points)
            # if this is a uncensored data point
            if e == 1:
                y_true_i[time_index] = 1
                y_true.append(y_true_i)
            # if this is a censored data point
            else:
                y_true_i[time_index:] = 1
                y_true.append(y_true_i)
        y_true = torch.Tensor(np.array(y_true))

        return y_true

    def get_time_points(self, T_max, extra_time_percent=0.1):
        """
        Get time points for the MTLR model
        """
        # Get time points in the time axis
        time_points = np.linspace(0, T_max * (1 + extra_time_percent), self.time_num + 1)

        return time_points

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
        anno = pd.read_csv('./anno/A_anno.txt', dtype={'CHR': str}, index_col=0, sep='\t')
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
        ratio = self.phases_data_ratios[self.phase]
        if self.train_in_phases and ratio != 1:
            np.random.seed(self.seed)
            train_index = np.random.choice(self.train_index, size=int(ratio * len(self.train_index)), replace=False)
            np.random.seed(self.seed)
            val_index = np.random.choice(self.val_index, size=int(ratio * len(self.val_index)), replace=False)
        else:
            train_index = self.train_index
            val_index = self.val_index

        if self.mode == 'pretraining':
            if not self.train_in_phases:
                if self.pretraining_data_ratio != 1:
                    np.random.seed(self.seed)
                    train_index = np.random.choice(self.train_index, size=int(self.pretraining_data_ratio * len(self.train_index)), replace=False)
                    np.random.seed(self.seed)
                    val_index = np.random.choice(self.val_index, size=int(self.pretraining_data_ratio * len(self.val_index)), replace=False)
            if stage == "fit" or stage is None:
                self.trainset = ABCDataset(self.A_df, self.B_df, self.C_df, train_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                self.valset = ABCDataset(self.A_df, self.B_df, self.C_df, val_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
            
            if stage == "test" or stage is None or stage == "predict":
                self.testset = ABCDataset(self.A_df, self.B_df, self.C_df, self.test_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
        
        elif self.mode == 'downstream':
            if not self.train_in_phases:
                if self.train_downstream_on_some_types:
                    # ds_tumour_indices = self.tumour_index['Index'][self.tumour_index['Tumour type'].isin(self.ds_cancer_types)]
                    # all_indices = np.arange(self.labels.shape[0])[self.labels_idxmax.isin(ds_tumour_indices)]
                    train_index = np.intersect1d(self.all_indices, self.train_index)
                    val_index = np.intersect1d(self.all_indices, self.val_index)
                    wandb.config.update({'num_ds_train_samples': train_index.shape[0], 'num_ds_val_samples': val_index.shape[0]}, allow_val_change=True)
                else:
                    if self.downstream_data_ratio != 1:
                        np.random.seed(self.seed)
                        train_index = np.random.choice(self.train_index, size=int(self.downstream_data_ratio * len(self.train_index)), replace=False)
                        np.random.seed(self.seed)
                        val_index = np.random.choice(self.val_index, size=int(self.downstream_data_ratio * len(self.val_index)), replace=False)
            if stage == "fit" or stage is None:
                if self.use_test_as_val_for_downstream:
                    self.trainset = ABCDataset(self.A_df, self.B_df, self.C_df, np.concatenate((train_index, val_index)), self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                    self.valset = ABCDataset(self.A_df, self.B_df, self.C_df, self.test_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                else:
                    self.trainset = ABCDataset(self.A_df, self.B_df, self.C_df, train_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                    self.valset = ABCDataset(self.A_df, self.B_df, self.C_df, val_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
            
            if stage == "test" or stage is None:
                self.testset = ABCDataset(self.A_df, self.B_df, self.C_df, self.test_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)

            if stage == "predict" or stage is None:
                if self.prediction_data == 'train':
                    if self.use_test_as_val_for_downstream:
                        self.predset = ABCDataset(self.A_df, self.B_df, self.C_df, np.concatenate((train_index, val_index)), self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                    else:
                        self.predset = ABCDataset(self.A_df, self.B_df, self.C_df, train_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                elif self.prediction_data == 'val':
                    if self.use_test_as_val_for_downstream:
                        self.predset = ABCDataset(self.A_df, self.B_df, self.C_df, self.test_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                    else:
                        self.predset = ABCDataset(self.A_df, self.B_df, self.C_df, val_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)
                elif self.prediction_data == 'test':
                    self.predset = ABCDataset(self.A_df, self.B_df, self.C_df, self.test_index, self.split_A, self.split_B, self.ds_tasks, self.labels, self.survival_T_array, self.survival_E_array, self.y_true_tensor, self.values)

    def teardown(self, stage=None):
        if hasattr(self, 'trainset'):
            del self.trainset
        if hasattr(self, 'valset'):
            del self.valset
        if hasattr(self, 'testset'):
            del self.testset
        if hasattr(self, 'predset'):
            del self.predset
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predset, batch_size=self.batch_size, num_workers=self.num_workers)
