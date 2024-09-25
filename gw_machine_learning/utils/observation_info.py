import csv
from random import sample
import h5py
import numpy as np
import os
from sympy import shape
from scipy.stats import shapiro
from scipy.stats import ks_2samp
from scipy.stats import anderson
from statsmodels.stats.diagnostic import lilliefors

output_file = "report/obs_statistics.csv"  # Replace with the desired output file path
if os.path.exists(output_file):
    os.remove(output_file)

# Define the field names for the CSV file
field_names = ["var", "number_of_obs", "min", "2.5th", "mean", "median", "std", "97.5th", "max", "Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "Lilliefors"]

# Create a list to store the statistics
statistics = []

for RESOLUTION in [250, 100, 50, 30]:
    path = f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5"
    with h5py.File(path, 'r') as f:
        dataset_names = [f'obs_H_COND_1_{RESOLUTION}m', f'obs_H_COND_2_{RESOLUTION}m',
                          f'obs_SWL_{RESOLUTION}m', f'obs_V_COND_1_{RESOLUTION}m',
                            f'obs_V_COND_2_{RESOLUTION}m', f'obs_TRANSMSV_1_{RESOLUTION}m',
                              f'obs_TRANSMSV_2_{RESOLUTION}m', f'obs_AQ_THK_1_{RESOLUTION}m', 
                              f'obs_AQ_THK_2_{RESOLUTION}m']

        for dataset_name in dataset_names:
            dataset = f[dataset_name][:]
            
            ## flatten the dataset
            dataset = dataset.flatten()
            ## remove -999 values
            dataset = dataset[dataset != -999]

            # Calculate the statistics
            number_of_obs = len(dataset)
            min_val = np.min(dataset)
            max_val = np.max(dataset)
            percentile_2_5 = np.percentile(dataset, 2.5)
            mean_val = np.mean(dataset)
            median_val = np.median(dataset)
            std_val = np.std(dataset)
            percentile_97_5 = np.percentile(dataset, 97.5)
            ## normality test for large dataset: Shapiro-Wilk test
            sampled_data = sample(list(dataset), 1000)     
            stat, p = shapiro(sampled_data)
            ## decide the normality
            alpha = 0.05
            if p > alpha:
                shape_test = "True"
            else:
                shape_test = "False"
            ## Kolmogorov-Smirnov test
            stat, p = ks_2samp(dataset, 'norm')
            ## decide the normality
            alpha = 0.05
            if p > alpha:
                ks_test = "True"
            else:
                ks_test = "False"
            # Anderson-Darling test
            result = anderson(dataset)
            if result.statistic < result.critical_values[2]:
                ad_test = "True"
            else:
                ad_test = "False"
            
            # Lilliefors test
            stat, p = lilliefors(dataset)
            ## decide the normality
            alpha = 0.05
            if p > alpha:
                lilliefors_test = "True"
            else:
                lilliefors_test = "False"

            # Add the statistics to the list
            statistics.append({
                "var": dataset_name,
                "number_of_obs": number_of_obs,
                "min": min_val,
                "2.5th": percentile_2_5,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "97.5th": percentile_97_5,
                "max": max_val,
                "Shapiro-Wilk": shape_test,
                "Kolmogorov-Smirnov": ks_test,
                "Anderson-Darling": ad_test,
                "Lilliefors": lilliefors_test
            })

# Sort the statistics by variable name
statistics.sort(key=lambda x: x["var"])

# Write the statistics to the CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(statistics)

print("Statistics written to file successfully.")
