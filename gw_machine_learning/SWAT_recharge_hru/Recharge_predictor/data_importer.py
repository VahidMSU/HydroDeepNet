
import pandas as pd
import numpy as np
import os
import logging

def print_features(numerical_features, categorical_features):
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    with open('results/numerical_features.txt', 'w') as f:
        for feature in numerical_features:
            f.write(f"{feature}\n")
    with open('results/categorical_features.txt', 'w') as f:
        for feature in categorical_features:
            f.write(f"{feature}\n")

def import_hru_data(file, max_samples=200000):
    # Load data
    df = pd.read_pickle(file)
    df['target'] = df['target'].astype(np.float32)
    df['target'] = np.where(df['target'] < 10, 0, df['target'])
    df['target'] = np.where((df['target'] >= 10) & (df['target'] < 20), 1, df['target'])
    df['target'] = np.where((df['target'] >= 20) & (df['target'] < 30), 2, df['target'])
    df['target'] = np.where((df['target'] >= 30) & (df['target'] < 40), 3, df['target'])
    df['target'] = np.where((df['target'] >= 40) & (df['target'] < 50), 4, df['target'])
    df['target'] = np.where((df['target'] >= 50) & (df['target'] < 100), 5, df['target'])
    df['target'] = np.where(df['target'] >= 100, 6, df['target'])
    df['target'] = df['target'].astype(int)

    # Sample the data to ensure the total number of samples is at most 1 million
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    numerical_features = ['model_name',
                          'area', 'lat', 'lon',
                          'elev', 'slp', 'slp_len',
                          'lat_len', 'cn_froz',
                          'perco', 'epco',
                          'esco', 'cn3_swf',
                          'tmp_lag', 'fall_tmp',
                          'melt_tmp', 'melt_min',
                          'melt_max', 'k',
                          'surq_lag',
                          'evap_co',
                          'hru_frac',
                          'month',
                          'year',
                          'precip', 
                          'lyr',
                        'dp', 
                        'bd', 
                        'awc', 
                        'soil_k',
                          'carbon',
                            'clay', 
                            'silt', 
    ]
    categorical_features = [
                            "lu_mgt",
                            "cc_name",
                            "vpu_id"]

    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    for feature in numerical_features:
        df[feature] = df[feature].astype(np.float32) if feature in df.columns else 0

    numerical_data = df[numerical_features]
    categorical_data = df[categorical_features]
    target = df['target']
    print(f"target shape: {target.shape}")
    print(f"numerical_data shape: {numerical_data.shape}")
    print(f"categorical_data shape: {categorical_data.shape}")

    return numerical_data, categorical_data, target