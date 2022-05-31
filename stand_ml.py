import pandas as pd
import numpy as np
from pycaret.classification import *

data = pd.read_csv('latent_space.tsv',sep='\t', header=0, index_col=0)
sample_list_path = 'data/ABC_inter/Normalized/sample_lists/class/ABC_inter/sample_list.tsv'
sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
labels_path = 'data/ABC_inter/Normalized/labels.tsv'
labels = pd.read_csv(labels_path, sep='\t', header=0, index_col=0)
labels = labels.loc[sample_list]

input_data = pd.concat([data, labels], axis=1)

exp = setup(data=input_data, target='sample_type.samples', use_gpu=True, fold_strategy='stratifiedkfold', fold=5, feature_selection=True, fix_imbalance=True, normalize=False)

best = compare_models()

predict_model(best)