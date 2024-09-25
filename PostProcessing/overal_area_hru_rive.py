
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    df_models = "model_characteristics/SWAT_gwflow_MODEL/df_models.csv"
    performance_df = "overal_best_performance/SWAT_gwflow_MODEL/best_performance_0000.csv"
    df_models = pd.read_csv(df_models)
    best_performance = pd.read_csv(performance_df)
    df_models = df_models[df_models['NAME'].isin(best_performance['NAME'])]
    ## number of unique NAME
    print(f"Number of unique NAME: {df_models['NAME'].nunique()}")
    ### get the total number of HRUs
    print(f"Total number of HRUs: {df_models['HRU_Count'].sum()}")
    ## get the total number of n_rivers
    print(f"Total number of n_rivers: {df_models['n_rivers'].sum()}")
    ## get the total number of Total_Area
    print(f"Total number of Total_Area: {df_models['Total_Area'].sum()}")
    ## get the total number of number_of_lakes
    print(f"Total number of number_of_lakes: {df_models['number_of_lakes'].sum()}")
    ## get the total size of total_Lake_area_sqkm
    print(f"Total size of total_Lake_area_sqkm: {df_models['total_Lake_area_sqkm'].sum()}")
