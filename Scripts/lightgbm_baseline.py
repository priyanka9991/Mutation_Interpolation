import os
import pandas as pd


def one_hot_encode_features(data):
    """
    Convert categorical features in a DataFrame to one-hot encoded format.
    
    Args:
    data (pd.DataFrame): DataFrame containing the categorical features.
    
    Returns:
    pd.DataFrame: DataFrame with one-hot encoded features.
    """
    return pd.get_dummies(data)

    #Response data
    res = pd.read_csv('../csa_data/raw_data/y_data/response.tsv', sep='\t', engine='c',
                      na_values=['na', '-', ''], header=0, index_col=None)

    
    print(res)
    one_hot_encode_features(res)