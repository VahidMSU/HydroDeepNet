import matplotlib.pyplot as plt
import flopy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
def plot_data(datasets, titles, model_input_figure_path, vmin=None, vmax=None,base_font_size = 10, figsize = (20, 15), dpi=300):
    # Set a base font size
    increased_font_size = base_font_size * 1.8  # Increase font size by 80%

    fig, axs = plt.subplots(3, 4, figsize = figsize)
    for ax, data, title in zip(axs.flat, datasets, titles):
        im = ax.imshow(data, vmin = vmin, vmax=vmax)
        ax.set_title(title, fontsize = increased_font_size)

        # Define the ticks to be at every 200th index value
        ax.set_xticks(np.arange(0, data.shape[1], 200))
        ax.set_yticks(np.arange(0, data.shape[0], 200))

        # Define the labels to be the index value (assuming index value corresponds to row/column number)
        ax.set_xticklabels(np.arange(0, data.shape[1], 200), rotation=90, fontsize=increased_font_size * 0.7)
        ax.set_yticklabels(np.arange(0, data.shape[0], 200), fontsize=increased_font_size * 0.7)

        # Set axis labels with increased font size
        ax.set_xlabel('Column', fontsize=increased_font_size)
        ax.set_ylabel('Row', fontsize=increased_font_size)

        # Add and configure colorbar with increased font size
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=increased_font_size * 0.7)

    plt.tight_layout()
    plt.savefig(model_input_figure_path, dpi=dpi)
    plt.close()



def river_images(swat_river):
    mask=swat_river!=0
    river_images=swat_river
    river_images[mask]=1
    return(river_images)

def plot_heads(LEVEL, NAME, RESOLUTION, MODEL_NAME, cmap = 'viridis', dpi = 300):
    """
    This function reads a MODFLOW head binary file and creates a plot for the head data for the last time step.

    Parameters:
    cmap (str): The colormap to use for the plot. Default is 'viridis'.

    Returns:
    None.
    """
    BASE_PATH = "{SWATGenXPaths.base_path}"
    # create the headfile object
    path = os.path.join(BASE_PATH,f'SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/',MODEL_NAME+'.hds')
    headobj = flopy.utils.binaryfile.HeadFile(path)

    # get all of the time steps
    times = headobj.get_times()

    # Get the head data for the last time
    head = headobj.get_data(totim=times[-1])

    # plot the heads for the last time step
    plt.figure(figsize=(10,10))
    mask = head[0, :, :] > 0
    masked_data = np.ma.masked_where(~mask, head[0, :, :])
    plt.imshow(masked_data, cmap='viridis') # Replace 'viridis' with your desired colormap
    # 0 is the first layer and then -1 is the last layer
    plt.colorbar(label='Head (meters)')
    plt.title('Heads for last time step')

    path = os.path.join(BASE_PATH, fr"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/head_of_last_time_step.jpeg")

    plt.savefig(path, dpi=dpi)

    plt.close()

    return head[0, :, :]





def create_plots_and_return_metrics (df_sim_obs, LEVEL, NAME,MODEL_NAME, dpi=300):
    # Calculate metrics

    df_sim_obs.dropna(subset=['obs_head_m','obs_SWL_m' ,'sim_head_m', 'obs_SWL_m'], inplace=True)

    print(f"Number of observations: {len(df_sim_obs.obs_SWL_m)}")
    print(f"Number of simulations: {len(df_sim_obs.sim_SWL_m)}")

    df_sim_obs=df_sim_obs[df_sim_obs.obs_SWL_m>0]
    df_sim_obs = df_sim_obs[(df_sim_obs.obs_SWL_m<df_sim_obs.obs_SWL_m.quantile(0.975))  &  (df_sim_obs.obs_SWL_m>df_sim_obs.obs_SWL_m.quantile(0.0245))  ]
    df_sim_obs = df_sim_obs[(df_sim_obs.sim_SWL_m<df_sim_obs.sim_SWL_m.quantile(0.975))  &  (df_sim_obs.sim_SWL_m>df_sim_obs.sim_SWL_m.quantile(0.0245))  ]
    max_ax= max(df_sim_obs.obs_SWL_m.max(), df_sim_obs.sim_SWL_m.max())
    min_ax= min(0,df_sim_obs.obs_SWL_m.min(), df_sim_obs.sim_SWL_m.min())
    TYPES = ['SWL','head']

    for TYPE in TYPES:
        obs = df_sim_obs[f'obs_{TYPE}_m'].values
        sim = df_sim_obs[f'sim_{TYPE}_m'].values
        nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
        mse = mean_squared_error(obs, sim)
        mae = mean_absolute_error(obs, sim)
        pbias = 100 * np.sum(obs - sim) / np.sum(obs)


        # KGE components
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        print('Number of observations: ', len(obs))
        print('Number of simulations: ', len(sim))
        # Create scatter plot
        plt.scatter(obs, sim, c='blue', marker='o', label='Data points')
        plt.xlabel(f'Observed {TYPE} (m)')
        plt.ylabel(f'Simulated {TYPE}(m)')
        plt.title(f'Observed vs Simulated {TYPE} (m)')

        # Add grid, legend, etc for a professional look
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Define the properties for the background box of the annotation
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="white", lw=0)

        # Add text annotations with background box for the metrics
        plt.annotate(f'NSE = {nse:.2f}', xy=(0.7, 0.1), xycoords='axes fraction', bbox=bbox_props)
        plt.annotate(f'MSE = {mse:.2f}', xy=(0.7, 0.15), xycoords='axes fraction', bbox=bbox_props)
        plt.annotate(f'MAE = {mae:.2f}', xy=(0.7, 0.2), xycoords='axes fraction', bbox=bbox_props)
        plt.annotate(f'PBIAS = {pbias:.2f}', xy=(0.7, 0.25), xycoords='axes fraction', bbox=bbox_props)
        plt.annotate(f'KGE = {kge:.2f}', xy=(0.7, 0.3), xycoords='axes fraction', bbox=bbox_props)
        model_output_figure_path = f"{SWATGenXPaths.base_path}SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/{TYPE}_simulated_figure.jpeg"

        plt.savefig(model_output_figure_path, dpi= dpi)
        plt.close()

    return nse, mse, mae, pbias, kge
