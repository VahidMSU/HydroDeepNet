import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

def day_to_month(day):
    """Convert day of the year to month."""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative_days = np.cumsum(days_in_month)
    return np.searchsorted(cumulative_days, day, side='right') + 1

def process_model(cc_model):
    TARGET_PATH = "E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/40500010102/climate_change_models/"

    print(f"Processing {cc_model}")
    cc_model_path = os.path.join(TARGET_PATH, cc_model)
    pcp_files = [file for file in os.listdir(cc_model_path) if file.endswith('.pcp')]

    if not pcp_files:
        return None

    all_pcps = []

    for pcp_file in pcp_files:
        pcp_data = pd.read_csv(os.path.join(cc_model_path, pcp_file), header=None, sep='\s+', skiprows=3, names=['year', 'day', 'pcp'])
        pcp_data['name'] = f"{pcp_file.split('.')[0]}_{cc_model}"
        pcp_data['month'] = pcp_data['day'].apply(day_to_month)
        all_pcps.append(pcp_data)

    if not all_pcps:
        return None

    all_pcps = pd.concat(all_pcps)
    all_pcps['year'] = all_pcps['year'].astype(int)
    all_pcps['day'] = all_pcps['day'].astype(int)
    all_pcps['pcp'] = all_pcps['pcp'].astype(float)

    # Sum daily precipitation for each month
    monthly_pcp = all_pcps.groupby(['year', 'month', 'name'])['pcp'].sum().reset_index()

    plt.figure()
    sns.lineplot(data=monthly_pcp, x='year', y='pcp', hue='month', errorbar="sd")
    plt.xlabel('Year')
    plt.ylabel('Total Monthly Precipitation (mm)')
    plt.title(f'Total Monthly Precipitation for {cc_model}')
    plt.savefig(os.path.join(cc_model_path, 'total_monthly_precipitation.png'))
    plt.close()

    return monthly_pcp

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

    # Calculate the 5th and 95th percentiles for each month
    percentiles = all_pcps_models.groupby(['year', 'month'])['pcp'].agg([
        ('5th', lambda x: np.percentile(x, 5)),
        ('25th', lambda x: np.percentile(x, 25)),
        ('50th', lambda x: np.percentile(x, 50)),
        ('75th', lambda x: np.percentile(x, 75)),
        ('95th', lambda x: np.percentile(x, 95))
    ]).reset_index()

    plt.figure(figsize=(14, 7))

    # Plot the percentiles with fill_between for each month
    for month in range(1, 13):
        month_data = percentiles[percentiles['month'] == month]
        plt.fill_between(month_data['year'], month_data['5th'], month_data['95th'], alpha=0.3, label=f'Month {month} (5th-95th)')
        plt.plot(month_data['year'], month_data['50th'], linestyle='-', label=f'Month {month} (50th)')

    plt.xlabel('Year')
    plt.ylabel('Total Monthly Precipitation (mm)')
    plt.title('Total Monthly Precipitation for All Models (5th to 95th Percentiles)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(TARGET_PATH, 'total_monthly_precipitation_all_models.jpeg'), dpi=300, bbox_inches='tight')
    plt.show()
