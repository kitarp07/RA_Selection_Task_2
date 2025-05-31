import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def load_data(path):
    """
    Load labeled dataset from a CSV file.
    
    Returns:
    - X: features
    - y: target labels
    - id: ID column
    """
    data = pd.read_csv(path)
    id = data['ID']
    X = data.drop(['ID', 'CLASS'], axis=1)
    y = data['CLASS']
    return X, y, id

def load_blinded_data(path):
    
    """
    Load unlabeled (blinded) dataset from a CSV file.
    
    Returns:
    - X: features
    - id: ID column
    """
    data = pd.read_csv(path)
    id = data['ID']
    X = data.drop(['ID'], axis=1)
    return X, id

def replace_inf_with_nan(X):
    """
    Replace infinite values (inf/-inf) with NaN in-place.
    """
    #replace infinity value 
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
def remove_almost_constant_features(X, threshold=0.01):
    """
    Remove features with variance below a threshold (i.e., almost constant).
    
    Returns:
    - DataFrame with low-variance features removed
    """
    # Remove features with variance below 0.01 (adjust as needed) - almost constant ones
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    mask = selector.get_support()
    selected_columns_vt = X.columns[mask]
    X_reduced_df = pd.DataFrame(X_reduced, columns=selected_columns_vt)
    return X_reduced_df

def fill_cols_with_na_val(X):
    """
    Fill missing values in specific known columns with their mean.
    
    Returns:
    - DataFrame with imputed values
    """
    # Impute with mean because there are low number of inf values
    X[['Feature_90', 'Feature_72']] = X[['Feature_90', 'Feature_72']].fillna(X[['Feature_90', 'Feature_72']].mean())
    
    return X

def remove_features_with_na_rows(X):
    """
    Drop features (columns) that  have high number of missing values.
    
    Returns:
    - DataFrame with such columns removed
    """
    missing_per_feature = X.isnull().sum()
    cols_to_drop = missing_per_feature[missing_per_feature > 0].index # get column names of data with missing values
    data_dropped_cols = X.drop(columns=cols_to_drop) #drop columns with missing values
    return data_dropped_cols


def remove_highly_correlated_features(X, threshold=0.9):
    """
    Remove features that are highly correlated with each other.
    
    Returns:
    - DataFrame with one feature from each highly correlated pair removed
    """
    # Compute correlation matrix (absolute value)
    corr_matrix = X.corr().abs()
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_from_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
    df_reduced = X.drop(columns=to_drop_from_corr)
    return df_reduced

def preprocess_data(X_train, X_test, X_blinded_test):
    """
    Apply full preprocessing pipeline:
    - Handle inf values
    - Remove low-variance features
    - Impute missing values
    - Drop remaining NaN columns
    - Remove highly correlated features
    
    Returns:
    - Processed versions of X_train, X_test, and X_blinded_test with same feature structure
    """
    replace_inf_with_nan(X_train)
    
    X_removed_constant_features = remove_almost_constant_features(X_train)
    
    X_fill_na = fill_cols_with_na_val(X_removed_constant_features)
    
    X_removed_na = remove_features_with_na_rows(X_fill_na)
    
    X_remove_highly_correlated_data = remove_highly_correlated_features(X_removed_na)
    X_test_filter_columns = X_test[X_remove_highly_correlated_data.columns]
    X_blinded_test_filter_columns = X_blinded_test[X_remove_highly_correlated_data.columns]
    
    return X_remove_highly_correlated_data, X_test_filter_columns, X_blinded_test_filter_columns


def scale_values(X_train, X_test, X_blinded_test):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Returns:
    - Scaled versions of training, test, and blinded test sets as DataFrames
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_blinded_test_scaled = scaler.transform(X_blinded_test)
    
    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_blinded_test_scaled = pd.DataFrame(X_blinded_test_scaled, columns=X_blinded_test.columns)
    return X_scaled, X_test_scaled, X_blinded_test_scaled
    
def select_k_best(X_train, y_train, X_test, X_blinded_test, k):
    """
    Univariate Feature Selection
    Select top-k features based on ANOVA F-value between feature and label.
    
    Returns:
    - X_train_selected: training features with top-k selection
    - X_test_selected: test features with same top-k
    - X_blinded_test_selected: blind test features with same top-k
    - selected_features: names of selected features
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected_array = selector.fit_transform(X_train, y_train)
    # Get boolean mask of selected features
    mask = selector.get_support()

    # Get the selected feature names from original DataFrame
    selected_features = X_train.columns[mask]

    # Convert numpy array back to DataFrame with selected columns
    X_train_selected = pd.DataFrame(X_selected_array, columns=selected_features)
    
    X_test_selected = X_test[X_train_selected.columns]
    X_blinded_test_selected = X_blinded_test[X_train_selected.columns]

    print(f"Selected {k} features")
    return X_train_selected, X_test_selected,X_blinded_test_selected, selected_features


    

        
# a = remove_almost_constant_features(X)
# b = fill_cols_with_na_val(a)
# c = remove_features_with_na_rows(b)
# d = remove_highly_correlated_features(c)
# e = scale_values(d)

# f, sf = select_k_best(e, 60)

# sf


