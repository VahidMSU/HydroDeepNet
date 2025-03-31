try:
	from performance_metrics import mape, nse, pbias
except Exception:
	from ModelProcessing.performance_metrics import mape, nse, pbias
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd
import glob
from datetime import datetime
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def annotating_scores(arg0, arg1, arg2):
    plt.annotate(f'MAPE: {arg0:.2f}', xy=(0.05, 0.85), xycoords='axes fraction')
    plt.annotate(f'NSE: {arg1:.2f}', xy=(0.05, 0.75), xycoords='axes fraction')
    plt.xlabel(arg2)


def get_figures(BASE_PATH, LEVEL, VPUID, NAME,MODEL_NAME):


    for time_step in ['daily', 'monthly']:
        stations = pd.DataFrame(columns=['station_name', 'objective_values', 'file'])
        files = glob.glob(os.path.join(BASE_PATH,f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/', f"figures_{MODEL_NAME}_calibration_{time_step}", "*.jpeg"))
        if len(files)>100:
            for i, file in enumerate(files):
                obj_value = float(os.path.basename(file).split('_')[0])
                station = os.path.basename(file).split('_')[2]

                stations.loc[i, 'station_name'] = station
                stations.loc[i, 'objective_values'] = obj_value
                stations.loc[i, 'file'] = file
            for station in stations['station_name'].unique():
                temp = stations[stations['station_name'] == station]
                fig_to_save = temp.sort_values(by='objective_values', ascending=False).iloc[0]['file']
                directory_path_si = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/calibration_figures_{MODEL_NAME}/')
                os.makedirs(directory_path_si, exist_ok=True)
                try:
                    shutil.copy2(fig_to_save, directory_path_si)
                except Exception as e:
                    logging.info(f"Problem with copying the image {e}")

def plot_domain(domain,fig_output_path):
    """
    Plot a domain array with specific colors:
    - Black for 0 (assuming 0 represents 'active')
    - Green for 1 (assuming 1 represents 'boundary')
    - Red for 2 (assuming 2 represents 'lakes')
    :param domain: A 2D numpy array representing the domain.
    """

    # Check unique values in the domain
    unique_values = np.unique(domain)
    logging.info(f'Unique values in domain: {unique_values}')

    # Define colors and bins based on unique values
    colors = ['blue', 'green', 'black']  # black for 0, green for 1, red for 2
    n_bins = [-0.5, 0.5, 1.5, 2.5]  # Bins for your data
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=len(colors))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the domain with the custom colormap
    cax = ax.imshow(domain, cmap=cmap, norm=mcolors.BoundaryNorm(n_bins, cmap.N))
    # Add a colorbar
    cbar = fig.colorbar(cax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['inactive', 'active domain', 'bound'])  # Vertically oriented color bar

    # Add title
    ax.set_title("Active Domain")

    plt.savefig(fig_output_path, dpi=400)
    plt.close()
