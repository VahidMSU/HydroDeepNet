import matplotlib.pyplot as plt
import flopy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from MODGenX.logger_singleton import get_logger

# Use the singleton logger pattern instead of directly initializing Logger
logger = get_logger()

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


def plot_modflow_sim_head(mf, new_model_ws, modflow_model_name, 
                         scenario_name, no_value, figure_directory):
    """
    Create comprehensive visualization of MODFLOW results including head distribution and observation comparisons.
    Also calculates and returns performance metrics for model evaluation.
    
    Parameters:
        mf (flopy.modflow.Modflow): MODFLOW model object
        new_model_ws (str): Model workspace directory path
        modflow_model_name (str): Name of the MODFLOW model
        scenario_name (str): Name for this model scenario
        no_value (float): Value to return on error
        figure_directory (str): Directory to save figures
        
    Returns:
        tuple: Performance metrics (nse, mse, mae, pbias, kge)
    """
    metrics_results = (no_value, no_value, no_value, no_value, no_value)  # Default return if errors occur
    head_path = os.path.join(new_model_ws, f'{modflow_model_name}.hds')
    
    try:
        if not os.path.exists(head_path) or os.path.getsize(head_path) == 0:
            logger.error(f"Head file not found or empty: {head_path}")
            return metrics_results
            
        # Load simulation output head file
        headobj = flopy.utils.binaryfile.HeadFile(head_path)
        sim_head = headobj.get_data(totim=headobj.get_times()[-1])
        
        # Process first layer head data - mask all no-value and unreasonable values
        first_head = sim_head[0,:,:]
        # Replace various no-data values with NaN for proper masking
        first_head[first_head == 9999] = np.nan
        first_head[first_head == -999] = np.nan
        first_head[first_head == -888.88] = np.nan
        # Also mask negative elevations if they represent no-data values in your context
        first_head[first_head < -100] = np.nan  # Assuming extremely negative values are invalid
        
        # Ensure figure directory exists
        os.makedirs(figure_directory, exist_ok=True)
        
        # Create a multi-panel figure for comprehensive visualization
        fig = plt.figure(figsize=(16, 10))
        
        # Head distribution map (top left) - using masked array for better visualization
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        # Create masked array to handle NaN values correctly
        masked_head = np.ma.masked_invalid(first_head)
        
        # Use a proper colormap with NaN transparency
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('white', alpha=0)  # Set NaN values to transparent white
        
        im = ax1.imshow(masked_head, cmap=cmap)
        ax1.set_title('Simulated Head Distribution', fontsize=14)
        plt.colorbar(im, ax=ax1, label='Head (m)')
        
        # Initialize empty metrics dictionary to store all calculated metrics
        all_metrics = {
            'sim_obs_metrics': {},  # Metrics from df_sim_obs data
            'hob_metrics': {}       # Metrics from HOB file
        }
        
        # STEP 1: Process observation HOB file if it exists
        hob_file = os.path.join(mf.model_ws, f"{modflow_model_name}.hob.out")
        if os.path.exists(hob_file):
            with open(hob_file) as file:
                lines = file.readlines()
                if len(lines) > 1:
                    # Extract simulated and observed values
                    sim = np.array([float(line.split()[0]) for line in lines[1:]])
                    obs = np.array([float(line.split()[1]) for line in lines[1:]])
                    
                    # Get mass balance information
                    lst_file = os.path.join(new_model_ws, f"{modflow_model_name}.list")
                    if os.path.exists(lst_file):
                        lst = flopy.utils.MfListBudget(lst_file)
                        CMB = lst.get_cumulative()
                        cmb_value = CMB['IN-OUT'][-1]
                        logger.info(f"Cumulative Mass Balance Error: {cmb_value}")
                        all_metrics['mass_balance_error'] = cmb_value
                    else:
                        cmb_value = np.nan
                    
                    # Remove outliers (same 1-99 percentile filter)
                    valid_indices = (
                        (sim > np.percentile(sim, 1)) & 
                        (sim < np.percentile(sim, 99)) & 
                        (obs > np.percentile(obs, 1)) & 
                        (obs < np.percentile(obs, 99))
                    )
                    sim_clean = sim[valid_indices]
                    obs_clean = obs[valid_indices]
                    
                    # Calculate performance metrics
                    if len(sim_clean) > 0:
                        # Calculate RMSE 
                        rmse = np.sqrt(np.mean((sim_clean - obs_clean)**2))
                        # Calculate NSE
                        nse = 1 - (np.sum((obs_clean - sim_clean)**2) / np.sum((obs_clean - np.mean(obs_clean))**2))
                        # Calculate MAE
                        mae = mean_absolute_error(obs_clean, sim_clean)
                        # Calculate R²
                        r2 = r2_score(obs_clean, sim_clean)
                        
                        # Store in metrics dict
                        all_metrics['hob_metrics'] = {
                            'rmse': rmse,
                            'nse': nse,
                            'mae': mae,
                            'r2': r2
                        }
                        
                        # Plot scatter on right side
                        ax2 = plt.subplot2grid((2, 3), (0, 2))
                        ax2.scatter(obs_clean, sim_clean, s=15, color='blue', alpha=0.7, edgecolor='k', linewidth=0.5)
                        
                        # Add 1:1 line
                        min_val = min(obs_clean.min(), sim_clean.min())
                        max_val = max(obs_clean.max(), sim_clean.max())
                        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='1:1 Line')
                        
                        ax2.set_xlabel('Observed Head (m)', fontsize=12)
                        ax2.set_ylabel('Simulated Head (m)', fontsize=12)
                        ax2.set_title('Observed vs. Simulated', fontsize=14)
                        ax2.grid(True, linestyle='--', alpha=0.7)
                        
                        # Create a text box with metrics
                        metrics_text = (
                            f"RMSE: {rmse:.2f} m\n"
                            f"NSE: {nse:.2f}\n"
                            f"MAE: {mae:.2f} m\n"
                            f"R²: {r2:.2f}\n"
                            f"Mass Balance Error: {cmb_value:.2e}"
                        )
                        
                        # Add metrics text box to the scatter plot
                        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                                fontsize=10, verticalalignment='top', bbox=props)
                        
                        # Histogram of errors (bottom)
                        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
                        errors = sim_clean - obs_clean
                        ax3.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                        ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
                        ax3.set_xlabel('Error (Simulated - Observed)', fontsize=12)
                        ax3.set_ylabel('Frequency', fontsize=12)
                        ax3.set_title('Distribution of Simulation Errors', fontsize=14)
                        
                        # Add error statistics to histogram
                        error_stats = (
                            f"Mean Error: {np.mean(errors):.2f} m\n"
                            f"Median Error: {np.median(errors):.2f} m\n"
                            f"Std Dev: {np.std(errors):.2f} m\n"
                            f"Points: {len(errors)}"
                        )
                        ax3.text(0.05, 0.95, error_stats, transform=ax3.transAxes,
                                fontsize=10, verticalalignment='top', bbox=props)
                    else:
                        logger.warning("No valid data points after filtering outliers")
                        default_layout_no_data(fig, "No valid data points after filtering outliers")
                else:
                    logger.warning("HOB file is empty")
                    default_layout_no_data(fig, "HOB file is empty")
        else:
            logger.warning(f"HOB file not found: {hob_file}")
            default_layout_no_data(fig, "HOB file not found")
     
        # Add overall title and adjust layout
        plt.suptitle(f'MODFLOW Simulation Results - {scenario_name}', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        figure_path = os.path.join(figure_directory, f'modflow_results_{scenario_name}.jpeg')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive visualization saved to {figure_path}")
        
        # Save metrics to CSV for reference
        metrics_path = os.path.join(figure_directory, f'metrics_{scenario_name}.csv')
        save_metrics_to_csv(all_metrics, metrics_path)
        
        return metrics_results

    except Exception as e:
        logger.error(f"Error in plot_modflow_sim_head: {str(e)}")
        return metrics_results

def default_layout_no_data(fig, message):
    """Helper function to create default plot layout when observation data is missing"""
    # Add a message in place of the scatter plot
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax2.text(0.5, 0.5, message, ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Observed vs. Simulated', fontsize=14)
    ax2.set_xlabel('Observed Head (m)', fontsize=12)
    ax2.set_ylabel('Simulated Head (m)', fontsize=12)
    
    # Add empty histogram
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    ax3.set_title('Distribution of Simulation Errors', fontsize=14)
    ax3.text(0.5, 0.5, "No error data available", ha='center', va='center', transform=ax3.transAxes)

def save_metrics_to_csv(metrics_dict, output_path):
    """Save metrics dictionary to a CSV file for reference"""
    # Flatten the nested dictionary for CSV output
    flat_metrics = {}
    
    for category, values in metrics_dict.items():
        if isinstance(values, dict):
            for metric_type, metric_values in values.items():
                if isinstance(metric_values, dict):
                    for metric_name, value in metric_values.items():
                        flat_metrics[f"{category}_{metric_type}_{metric_name}"] = value
                else:
                    flat_metrics[f"{category}_{metric_type}"] = metric_values
        else:
            flat_metrics[category] = values
    
    # Convert to DataFrame and save
    df = pd.DataFrame([flat_metrics])
    df.to_csv(output_path, index=False)
    logger.info(f"Metrics saved to {output_path}")