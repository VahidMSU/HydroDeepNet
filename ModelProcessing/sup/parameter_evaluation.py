import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def group_by_high_low_metrics(all_df, parameters_df):
    # Concatenating all dataframes
    print(f"columns: {all_df.columns}")
    for metric in ['RMSE', 'NSE', 'KGE', 'MPE']:
        import numpy as np
        all_df['station'] = np.where(all_df['station'].isin(['ET', 'Head', 'SWL', 'Yield']), all_df['station'], 'SF')
        objectives = all_df.station.unique()
        print(f"Objectives: {objectives}")  
        for objective in objectives:
            df = all_df.copy()
            df.reset_index(drop=True, inplace=True)
            
            # Grouping by NAME and analyzing RMSE within each group
            grouped = df[df.station == objective].groupby('NAME')
            ### print range of ET RMSE values
            print(f"Range of {objective} {metric} values: {grouped[metric].min().min()} - {grouped[metric].max().max()}")
            high_rmse_params_list = []
            low_rmse_params_list = []

            ## print the name with the highest RMSE
            print(f"NAME with highest NSE value: {grouped[metric].mean().idxmax()}")
            for name, group in grouped:
                high_rmse_threshold = group[metric].mean() - group[metric].std()  # or any threshold you want
                high_rmse_params = group[group[metric] > high_rmse_threshold]
                low_rmse_params = group[group[metric] <= high_rmse_threshold]
                
                high_rmse_params_list.append(high_rmse_params)
                low_rmse_params_list.append(low_rmse_params)
            
            high_rmse_params = pd.concat(high_rmse_params_list)
            low_rmse_params = pd.concat(low_rmse_params_list)

            #print(f"Parameters leading to high RMSE:")
            #print(high_rmse_params[['key', 'NAME', metric] + parameters_df.columns.tolist()])  # Display key, NAME, RMSE, and parameters
            
            # Plotting parameter distributions for high and low RMSE across different NAME groups
            parameter_cols = parameters_df.columns.tolist()[1:]  # Exclude 'key'
            num_params = len(parameter_cols)
            num_cols = 5
            num_rows = (num_params + num_cols - 1) // num_cols

            plt.figure(figsize=(15, num_rows * 5))
            for i, param in enumerate(parameter_cols):
                plt.subplot(num_rows, num_cols, i + 1)
                sns.boxplot(data=pd.concat([
                    high_rmse_params[[param]].assign(Group='High error'),
                    low_rmse_params[[param]].assign(Group='Low error')
                ]), x='Group', y=param)
                plt.title(param)
                plt.tight_layout()
            plt.suptitle(f'Parameter Distributions for High vs Low {objective} RMSE Grouped by NAME', y=1.02)
            plt.savefig(f"{objective}_figs_grouped_{metric}.png")
            plt.close()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def group_by_high_low_variance(all_df, parameters_df):
    """
    Group data by high and low variance of the error metrics across different objectives.
    
    Args:
    - all_df: DataFrame containing the metrics and parameters.
    - parameters_df: DataFrame containing parameter names.
    
    This function calculates variance for each group based on different objectives, and then 
    groups the parameters into high-variance and low-variance categories.
    """
    
    print(f"Columns: {all_df.columns}")
    
    # Loop over each metric
    for metric in ['RMSE', 'NSE', 'KGE', 'MPE']:
        
        # Grouping station categories: ET, Head, SWL, Yield, SF
        all_df['station'] = np.where(all_df['station'].isin(['ET', 'Head', 'SWL', 'Yield']), all_df['station'], 'SF')
        objectives = all_df.station.unique()
        print(f"Objectives: {objectives}")
        
        # Loop over each objective
        for objective in objectives:
            df = all_df.copy()
            df.reset_index(drop=True, inplace=True)
            
            # Group by 'NAME' and compute variance for each group
            grouped = df[df.station == objective].groupby('NAME')
            
            ### Print range of objective variance values
            print(f"Variance range for {objective} {metric}:")
            variance_df = grouped[metric].var().reset_index()
            print(variance_df[metric].min(), variance_df[metric].max())
            
            high_variance_params_list = []
            low_variance_params_list = []
            
            # Use the mean variance as the threshold to split groups
            variance_threshold = variance_df[metric].mean()
            print(f"Variance threshold for {objective} {metric}: {variance_threshold}")

            # Iterate over the groups
            for name, group in grouped:
                variance = group[metric].var()
                
                if variance > variance_threshold:
                    # High variance group
                    high_variance_params_list.append(group)
                else:
                    # Low variance group
                    low_variance_params_list.append(group)
            
            # Concatenate all high and low variance groups
            high_variance_params = pd.concat(high_variance_params_list)
            low_variance_params = pd.concat(low_variance_params_list)
            
            # Plot parameter distributions for high and low variance groups
            parameter_cols = parameters_df.columns.tolist()[1:]  # Exclude 'key'
            num_params = len(parameter_cols)
            num_cols = 5
            num_rows = (num_params + num_cols - 1) // num_cols
            
            plt.figure(figsize=(15, num_rows * 5))
            for i, param in enumerate(parameter_cols):
                plt.subplot(num_rows, num_cols, i + 1)
                sns.boxplot(data=pd.concat([
                    high_variance_params[[param]].assign(Group='High Variance'),
                    low_variance_params[[param]].assign(Group='Low Variance')
                ]), x='Group', y=param)
                plt.title(param)
                plt.tight_layout()
            
            plt.suptitle(f'Parameter Distributions for High vs Low Variance in {objective} {metric} Grouped by NAME', y=1.02)
            plt.savefig(f"{objective}_figs_grouped_variance_{metric}.png")
            plt.close()

    return high_variance_params, low_variance_params


def read_parameter_performance_files(BASE_PATH, NAMES, parameters_name, performance_name):
    all_df = []
    for NAME in NAMES:
        if os.path.exists(f"{BASE_PATH}/{NAME}/{parameters_name}") and os.path.exists(f"{BASE_PATH}/{NAME}/{performance_name}"):
           # print(f"Processing {NAME}")
            print(f"################### Path: {BASE_PATH}/{NAME} #################")
            parameters_df = pd.read_csv(f"{BASE_PATH}/{NAME}/{parameters_name}", engine='python', sep='\s+')
            performance_df = pd.read_csv(f"{BASE_PATH}/{NAME}/{performance_name}", sep='\t', engine='python')
            assert "key" in parameters_df.columns, "key column not found in parameters_df"    
            assert "key" in performance_df.columns, "key column not found in performance_df"
            df = parameters_df.merge(performance_df, on="key", how="inner")
            df['NAME'] = NAME
            all_df.append(df)
    all_df = pd.concat(all_df)
    return all_df, parameters_df


BASE_PATH = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12"
NAMES = os.listdir(BASE_PATH)
NAMES.remove("log.txt")
parameters_name = "CentralParameters.txt"
performance_name = "CentralPerformance.txt"

all_df, parameters_df = read_parameter_performance_files(BASE_PATH, NAMES, parameters_name, performance_name)
print(all_df.head())    
#group_by_high_low_metrics(all_df, parameters_df)
#group_by_high_low_variance(all_df, parameters_df)