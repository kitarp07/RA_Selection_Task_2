import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def load_data(path):
    """Load dataset and return X, y."""
    data = pd.read_csv(path)
    X = data.drop(['ID', 'CLASS'], axis=1)
    y = data['CLASS']
    return X, y

def replace_inf_with_nan(X):
    #replace infinity value 
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
def remove_almost_constant_features(X, threshold=0.01):
    # Remove features with variance below 0.01 (adjust as needed) - almost constant ones
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    mask = selector.get_support()
    selected_columns_vt = X.columns[mask]
    X_reduced_df = pd.DataFrame(X_reduced, columns=selected_columns_vt)
    return X_reduced_df

def fill_cols_with_na_val(X):
    # Impute with mean because there are low number of inf values
    X[['Feature_90', 'Feature_72']] = X[['Feature_90', 'Feature_72']].fillna(X[['Feature_90', 'Feature_72']].mean())
    
    return X

def remove_features_with_na_rows(X):
    missing_per_feature = X.isnull().sum()
    cols_to_drop = missing_per_feature[missing_per_feature > 0].index # get column names of data with missing values
    data_dropped_cols = X.drop(columns=cols_to_drop) #drop columns with missing values
    return data_dropped_cols

def remove_highly_correlated_features(X, threshold=0.9):
    # Compute correlation matrix (absolute value)
    corr_matrix = X.corr().abs()
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_from_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
    df_reduced = X.drop(columns=to_drop_from_corr)
    return df_reduced

def scale_values(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled
    
def select_k_best(X, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected_array = selector.fit_transform(X, y)
    # Get boolean mask of selected features
    mask = selector.get_support()

    # Get the selected feature names from original DataFrame
    selected_features = X.columns[mask]

    # Convert numpy array back to DataFrame with selected columns
    X_selected = pd.DataFrame(X_selected_array, columns=selected_features)

    print(f"Selected {k} features")
    return X_selected, selected_features
    

        
# a = remove_almost_constant_features(X)
# b = fill_cols_with_na_val(a)
# c = remove_features_with_na_rows(b)
# d = remove_highly_correlated_features(c)
# e = scale_values(d)

# f, sf = select_k_best(e, 60)

# sf


