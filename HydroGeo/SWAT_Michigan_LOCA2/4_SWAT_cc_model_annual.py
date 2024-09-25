import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

def process_model(cc_model):

    TARGET_PATH = "E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/40500010102/climate_change_models/"

    scenario = cc_model.split('_')[1]
    ensemble = cc_model.split('_')[2]


    print(f"Processing {cc_model} for scenario {scenario}, ensemble {ensemble}")
    cc_model_path = os.path.join(TARGET_PATH, cc_model)
    pcp_files = [file for file in os.listdir(cc_model_path) if file.endswith('.pcp')]

    if not pcp_files:
        return None

    all_pcps = []

    for pcp_file in pcp_files:
        pcp_data = pd.read_csv(os.path.join(cc_model_path, pcp_file), header=None, sep='\s+', skiprows=3, names=['year', 'day', 'pcp'])
        pcp_data['name'] = f"{pcp_file.split('.')[0]}_{cc_model}"
        pcp_data['scenario'] = scenario
        pcp_data['ensemble'] = ensemble
        all_pcps.append(pcp_data)

    if not all_pcps:
        return None

    all_pcps = pd.concat(all_pcps)
    all_pcps['year'] = all_pcps['year'].astype(int)
    all_pcps['day'] = all_pcps['day'].astype(int)
    all_pcps['pcp'] = all_pcps['pcp'].astype(float)

    return all_pcps.groupby(['year', 'name', 'scenario', 'ensemble'])['pcp'].sum().reset_index()

if __name__ == "__main__":
    # Paths
    BASE_PATH = "/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12"
    NAMES = os.listdir(BASE_PATH)
    NAMES.remove('log.txt')

    TARGET_PATH = "E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/40500010102/climate_change_models/"
    cc_models = [d for d in os.listdir(TARGET_PATH) if os.path.isdir(os.path.join(TARGET_PATH, d))]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_model, cc_models))

    # Filter out None results
    all_pcps_models = [result for result in results if result is not None]
    all_pcps_models = pd.concat(all_pcps_models)

    # Calculate the percentiles for each scenario
    scenarios = ['historical', 'ssp585', 'ssp245', 'ssp370']
    for scenario in scenarios:
        scenario_data = all_pcps_models[all_pcps_models['scenario'] == scenario]

        percentiles = scenario_data.groupby('year')['pcp'].agg([
            ('2.5th', lambda x: np.percentile(x, 2.5)),
            ('25th', lambda x: np.percentile(x, 25)),
            ('50th', lambda x: np.percentile(x, 50)),
            ('75th', lambda x: np.percentile(x, 75)),
            ('97.5th', lambda x: np.percentile(x, 97.5))
        ])

        plt.figure(figsize=(14, 7))
        plt.fill_between(percentiles.index, percentiles['2.5th'], percentiles['97.5th'], color='skyblue', alpha=0.5)
        plt.plot(percentiles.index, percentiles['2.5th'], color='red', alpha=0.7, linestyle='--')
        plt.plot(percentiles.index, percentiles['97.5th'], color='red', alpha=0.7, linestyle='--')
        plt.plot(percentiles.index, percentiles['25th'], color='green', alpha=0.7, linestyle='-.')
        plt.plot(percentiles.index, percentiles['75th'], color='green', alpha=0.7, linestyle='-.')
        plt.plot(percentiles.index, percentiles['50th'], color='blue', alpha=0.7, linestyle='-')

        plt.xlabel('Year')
        plt.ylabel('Total Annual Precipitation (mm)')
        plt.title(f'Total Annual Precipitation for Scenario {scenario}')
        plt.savefig(os.path.join(TARGET_PATH, f'total_annual_precipitation_{scenario}.jpeg'), dpi=300)
        plt.show()

    # Plot individual models and ensemble members for historical scenario
    historical_data = all_pcps_models[all_pcps_models['scenario'] == 'historical']
    unique_models = historical_data['name'].unique()

    for model in unique_models:
        model_data = historical_data[historical_data['name'] == model]
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=model_data, x='year', y='pcp', hue='ensemble')
        plt.xlabel('Year')
        plt.ylabel('Total Annual Precipitation (mm)')
        plt.title(f'Total Annual Precipitation for {model}')
        plt.savefig(os.path.join(TARGET_PATH, f'total_annual_precipitation_{model}.jpeg'), dpi=300)
        plt.show()
