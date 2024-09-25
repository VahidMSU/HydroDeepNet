import os
import numpy as np
import h5py
import pandas as pd
from sympy import flatten
import concurrent.futures

import matplotlib.pyplot as plt
RESOLUTIONS = [250, 30]

def process_target_arrays(RESOLUTION):
    path = f"/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5"
    print(path)

    target_arrays = [
        f'obs_H_COND_1_{RESOLUTION}m',
        f'obs_H_COND_2_{RESOLUTION}m',
        f'obs_SWL_{RESOLUTION}m',
        f'obs_V_COND_1_{RESOLUTION}m',
        f'obs_V_COND_2_{RESOLUTION}m',
        f'obs_TRANSMSV_1_{RESOLUTION}m',
        f'obs_TRANSMSV_2_{RESOLUTION}m',
        f'obs_AQ_THK_1_{RESOLUTION}m',
        f'obs_AQ_THK_2_{RESOLUTION}m'
    ]

    performance_data = []  # List to store performance data

    for target_array in target_arrays:
        simulation = f'kriging_output_{target_array.split("obs_")[1]}'
        
        with h5py.File(path, 'r') as f:
            print(target_array.split("obs_")[1].split("_250m")[0])
            simulation = f[simulation][:]
            target = f[target_array][:]
            flatten_target = list(flatten(target))
            flatten_simulation = list(flatten(simulation))
            # Replace -999 with nan
            flatten_target = [np.nan if x == -999 else x for x in flatten_target]
            flatten_simulation = [np.nan if x == -999 else x for x in flatten_simulation]
            # Drop nan values
            flatten_target = np.array(flatten_target)
            flatten_simulation = np.array(flatten_simulation)
            mask = ~np.isnan(flatten_target) & ~np.isnan(flatten_simulation)
            flatten_target = flatten_target[mask]
            flatten_simulation = flatten_simulation[mask]
            # Calculate the MSE, RMSE, NSE, R2, MPE
            mse = np.mean((flatten_target - flatten_simulation)**2)
            rmse = np.sqrt(mse)
            nse = 1 - np.sum((flatten_target - flatten_simulation)**2) / np.sum((flatten_target - np.mean(flatten_target))**2)
            r2 = 1 - np.sum((flatten_target - flatten_simulation)**2) / np.sum((flatten_target - np.mean(flatten_target))**2)
            mpe = np.mean((flatten_target - flatten_simulation) / flatten_target)
            print(f"{target_array}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, NSE: {nse:.2f}, R2: {r2:.2f}, MPE: {mpe:.2f}")

            # Append performance data to the list
            performance_data.append({
                'Target Array': target_array.split("obs_")[1].split("_250m")[0],
                "resolution": f"{RESOLUTION}m",
                'MSE': mse,
                'RMSE': rmse,
                'NSE': nse,
                'R2': r2,
                'MPE': mpe
            })

    return performance_data

def main():
    all_performance_data = []

    with concurrent.futures.ProcessPoolExecutor(50) as executor:
        results = executor.map(process_target_arrays, RESOLUTIONS)
        for result in results:
            all_performance_data.extend(result)

    # Create a DataFrame from the performance data
    df = pd.DataFrame(all_performance_data)

    # Write the DataFrame to a CSV file
    df.to_csv('report/EBK_performance.csv', index=False)

if __name__ == "__main__":
    main()
