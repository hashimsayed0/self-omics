import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_selection import SelectFpr, SelectKBest, chi2, f_classif, mutual_info_classif, SelectPercentile

def select_features(df, labels, train_id, val_id, test_id, feature_selection, feature_selection_alpha, feature_selection_percentile):
    X_train = df.iloc[:,train_id].T.values
    X_test = df.iloc[:,test_id].T.values
    X_val = df.iloc[:,val_id].T.values
    y_train = labels.iloc[train_id].values.ravel()

    if feature_selection == "f_test":
        # selector = SelectFpr(f_classif, alpha=feature_selection_alpha)
        selector = SelectPercentile(f_classif, percentile=feature_selection_percentile)
    elif feature_selection == "chi2":
        selector = SelectFpr(chi2, alpha=feature_selection_alpha)
    elif feature_selection == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=125)
    elif feature_selection == "all":
        ft = SelectFpr(f_classif, alpha=feature_selection_alpha)
        X_train_1 = ft.fit_transform(X_train, y_train)
        X_test_1 = ft.transform(X_test)
        chi = SelectFpr(chi2, alpha=feature_selection_alpha)
        X_train_2 = chi.fit_transform(X_train, y_train)
        X_test_2 = chi.transform(X_test)
        mi = SelectKBest(mutual_info_classif, k=125)
        X_train_3 = mi.fit_transform(X_train, y_train)
        X_test_3 = mi.transform(X_test)

        # take intersection of all three sets
        ft_mask = ft.get_support()
        chi_mask = chi.get_support()
        mi_mask = mi.get_support()
        # select indices of ft_mask that are true
        ft_indices = [i for i, x in enumerate(ft_mask) if x]
        # select indices of chi_mask that are true
        chi_indices = [i for i, x in enumerate(chi_mask) if x]
        # select indices of mi_mask that are true
        mi_indices = [i for i, x in enumerate(mi_mask) if x]
        # take intersection of all three sets
        indices = list(set(ft_indices) & set(chi_indices) & set(mi_indices))
        X_train = X_train[:, indices]
        X_test = X_test[:, indices]
        X_val = X_val[:, indices]

    # if feature_selection == "prior":
    #     features = ['ABCC9', 'ANKLE2', 'ANKRD13D', 'ANKZF1', 'AP1G2', 'ARRDC2', 'ATAD2', 'ATG4B', 'ATIC', 'BARD1', 'BCL6', 'BMS1', 'BRAP', 'BTG2', 'CBX5', 'CD2AP', 'CEBPZ', 'CIRBP', 'CREBL2', 'CRK', 'CYB5A', 'CYBRD1', 'DCAF13',
    #         'DCLRE1C', 'DDX42', 'DDX51', 'DENR', 'DHCR24', 'DNAJB6', 'DNPEP', 'DPYSL2', 'DROSHA', 'EMP2', 'ENY2', 'ERC1', 'ERCC6', 'ETNK1', 'FAM122B', 'FAM20B', 'FANCL', 'FOXK2', 'FRMD4B', 'GAK', 'GCN1L1', 'GGA3', 'GLUD1',
    #         'GMDS', 'GTPBP2', 'GTPBP4', 'HIPK3', 'HIST1H4B', 'HIST1H4C', 'IL6ST', 'KIAA0368', 'LAMB2', 'LOC100133091', 'LPAR6', 'LRPPRC', 'MAPK14', 'MAT2A', 'MBOAT2', 'MCM7', 'MYO10', 'NAA25', 'NOP56',
    #         'PTGES3', 'RBMS3', 'RICTOR', 'SNORA37', 'SPG7', 'TXNIP']
    #     X_train = X_train[:, [df.columns.get_loc(f) for f in features]]
    #     X_test = X_test[:, [df.columns.get_loc(f) for f in features]]
    #     X_val = X_val[:, [df.columns.get_loc(f) for f in features]]

    if feature_selection != "all":
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        X_val = selector.transform(X_val)
    
    n_features = X_train.shape[1]

    df = pd.DataFrame(index=np.arange(n_features), columns=df.columns)
    df.iloc[:,train_id] = X_train.T
    df.iloc[:,val_id] = X_val.T
    df.iloc[:,test_id] = X_test.T
                
    return df

def scale_features(scaler, df, train_id, val_id, test_id):
    X_train = df.iloc[:,train_id].T.values
    X_test = df.iloc[:,test_id].T.values
    X_val = df.iloc[:,val_id].T.values
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    scaled_df = pd.DataFrame(index=df.index, columns=df.columns)
    scaled_df.iloc[:,train_id] = X_train.T
    scaled_df.iloc[:,val_id] = X_val.T
    scaled_df.iloc[:,test_id] = X_test.T
    return scaled_df
