import matplotlib.pyplot as plt
import flopy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from MODGenX.Logger import Logger

logger = Logger(verbose=True)

# Modern matplotlib configuration for better visualization
def setup_matplotlib_style():
    """Configure matplotlib for better visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')  # Use a modern style
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
    })
    
# Call this at the module level
setup_matplotlib_style()

# Data preparation functions
def river_images(swat_river):
    """
    Convert river data to binary (0/1) representation.
    
    This function doesn't appear to be used anywhere in the codebase.
    The river preprocessing is handled differently in river_correction()
    and other functions.
    """
    mask = swat_river != 0
    river_images = swat_river.copy()  # Create a copy to avoid modifying the original
    river_images[mask] = 1
    return river_images


# Visualization functions
def plot_data(datasets, titles, model_input_figure_path, vmin=None, vmax=None, 
              base_font_size=10, figsize=(20, 15), dpi=300):
    """
    Create a multi-panel plot with customized formatting for spatial data visualization.
    
    Parameters:
        datasets (list): List of arrays to be plotted
        titles (list): List of titles for each subplot
        model_input_figure_path (str): Path to save the figure
        vmin (float, optional): Minimum value for color scaling
        vmax (float, optional): Maximum value for color scaling
        base_font_size (int, optional): Base font size for plot text
        figsize (tuple, optional): Figure size (width, height) in inches
        dpi (int, optional): Resolution for saved figure
    """
    # Use a context manager for figure to ensure proper cleanup
    with plt.style.context('seaborn-v0_8-whitegrid'):
        # Create figure with constrained_layout for better spacing
        fig, axs = plt.subplots(3, 4, figsize=figsize, constrained_layout=True)
        
        for ax, data, title in zip(axs.flat, datasets, titles):
            # Create a masked array for better NoData handling
            masked_data = np.ma.masked_where(data == 9999, data)
            im = ax.imshow(masked_data, vmin=vmin, vmax=vmax, cmap='viridis')
            
            # More modern title layout
            ax.set_title(title, fontsize=base_font_size*1.8, pad=10)
            
            # Define ticks
            ax.set_xticks(np.arange(0, data.shape[1], 200))
            ax.set_yticks(np.arange(0, data.shape[0], 200))
            
            # Add more aesthetic customization
            ax.tick_params(
                axis='both', 
                which='both', 
                labelsize=base_font_size*0.7, 
                grid_alpha=0.3
            )
            
            # Use a specialized formatter for the colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=base_font_size*0.7)
        
        # Remove any unused subplots
        for ax in axs.flat[len(datasets):]:
            ax.remove()
            
        # Save with bbox_inches='tight' to avoid cutoff
        plt.savefig(model_input_figure_path, dpi=dpi, bbox_inches='tight')
        plt.close()


def plot_heads(username, VPUID, LEVEL, NAME, RESOLUTION, MODEL_NAME, cmap='viridis', dpi=300, path_handler=None):
    """
    Read a MODFLOW head binary file and create a plot for the head data for the last time step.

    Parameters:
        username (str): Username for file path construction
        VPUID (str): Virtual Polygon Unit ID
        LEVEL (str): Level information
        NAME (str): Name information
        RESOLUTION (str): Resolution information
        MODEL_NAME (str): Model name
        cmap (str, optional): Colormap to use for the plot
        dpi (int, optional): Resolution for saved figure
        path_handler (PathHandler, optional): Path handler for file paths
        
    Returns:
        numpy.ndarray: Head data for the last time step
    """
    # Use path_handler if provided, otherwise construct paths manually
    if path_handler:
        model_dir = path_handler.get_model_path()
        head_file_path = path_handler.get_output_file(MODEL_NAME + '.hds')
        output_path = path_handler.get_output_file("head_of_last_time_step.jpeg")
    else:
        BASE_PATH = f"/data/SWATGenXApp/Users/{username}/"
        model_dir = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/')
        head_file_path = os.path.join(model_dir, MODEL_NAME + '.hds')
        output_path = os.path.join(model_dir, "head_of_last_time_step.jpeg")
    
    try:
        headobj = flopy.utils.binaryfile.HeadFile(head_file_path)
        
        # Get all of the time steps
        times = headobj.get_times()

        # Get the head data for the last time
        head = headobj.get_data(totim=times[-1])

        # Plot the heads for the last time step
        plt.figure(figsize=(10, 10))
        mask = head[0, :, :] > 0
        masked_data = np.ma.masked_where(~mask, head[0, :, :])
        plt.imshow(masked_data, cmap=cmap)
        plt.colorbar(label='Head (meters)')
        plt.title('Heads for last time step')

        plt.savefig(output_path, dpi=dpi)
        plt.close()

        return head[0, :, :]
    
    except Exception as e:
        logger.error(f"Error in plot_heads: {e}")
        return None


# Metrics calculation and plotting functions
def calculate_performance_metrics(obs, sim):
    """
    Calculate various performance metrics between observed and simulated data.
    
    Parameters:
        obs (numpy.ndarray): Observed values
        sim (numpy.ndarray): Simulated values
        
    Returns:
        tuple: NSE, MSE, MAE, PBIAS, KGE
    """
    # Nash-Sutcliffe efficiency
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    
    # Mean squared error and mean absolute error
    mse = mean_squared_error(obs, sim)
    mae = mean_absolute_error(obs, sim)
    
    # Percent bias
    pbias = 100 * np.sum(obs - sim) / np.sum(obs)

    # Kling-Gupta Efficiency components
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return nse, mse, mae, pbias, kge


def create_plots_and_return_metrics(df_sim_obs, username, VPUID, LEVEL, NAME, MODEL_NAME, dpi=300, path_handler=None):
    """
    Create comparison plots between observed and simulated data and calculate performance metrics.
    
    Parameters:
        df_sim_obs (pandas.DataFrame): DataFrame containing observed and simulated values
        username (str): Username for file path construction
        VPUID (str): Virtual Polygon Unit ID
        LEVEL (str): Level information
        NAME (str): Name information
        MODEL_NAME (str): Model name
        dpi (int, optional): Resolution for saved figure
        path_handler (PathHandler, optional): Path handler for file paths
        
    Returns:
        tuple: NSE, MSE, MAE, PBIAS, KGE for the last data type processed
    """
    # Clean data by removing NaN values
    df_sim_obs.dropna(subset=['obs_head_m', 'obs_SWL_m', 'sim_head_m', 'sim_SWL_m'], inplace=True)

    print(f"Number of observations: {len(df_sim_obs.obs_SWL_m)}")
    print(f"Number of simulations: {len(df_sim_obs.sim_SWL_m)}")

    # Filter data to remove outliers
    df_sim_obs = df_sim_obs[df_sim_obs.obs_SWL_m > 0]
    df_sim_obs = df_sim_obs[(df_sim_obs.obs_SWL_m < df_sim_obs.obs_SWL_m.quantile(0.975)) & 
                           (df_sim_obs.obs_SWL_m > df_sim_obs.obs_SWL_m.quantile(0.0245))]
    df_sim_obs = df_sim_obs[(df_sim_obs.sim_SWL_m < df_sim_obs.sim_SWL_m.quantile(0.975)) & 
                           (df_sim_obs.sim_SWL_m > df_sim_obs.sim_SWL_m.quantile(0.0245))]
    
    metrics_results = None
    TYPES = ['SWL', 'head']

    for TYPE in TYPES:
        obs = df_sim_obs[f'obs_{TYPE}_m'].values
        sim = df_sim_obs[f'sim_{TYPE}_m'].values
        
        print(f'Number of {TYPE} observations: {len(obs)}')
        print(f'Number of {TYPE} simulations: {len(sim)}')
        
        # Calculate performance metrics
        nse, mse, mae, pbias, kge = calculate_performance_metrics(obs, sim)
        metrics_results = (nse, mse, mae, pbias, kge)
        
        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(obs, sim, c='blue', marker='o', label='Data points')
        plt.xlabel(f'Observed {TYPE} (m)')
        plt.ylabel(f'Simulated {TYPE} (m)')
        plt.title(f'Observed vs Simulated {TYPE} (m)')

        # Add grid, legend, etc.
        plt.grid(True)
        plt.legend()
        
        # Add 1:1 line for reference
        min_val = min(plt.xlim()[0], plt.ylim()[0])
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
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
        
        # Save the figure
        if path_handler:
            output_path = path_handler.get_output_file(f"{TYPE}_simulated_figure.jpeg")
        else:
            output_dir = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/"
            output_path = os.path.join(output_dir, f"{TYPE}_simulated_figure.jpeg")
        
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    # Return the last set of metrics calculated
    return metrics_results
