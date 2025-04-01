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
import time
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from ModelProcessing.logging_utils import get_logger

# Create module-level logger
logger = get_logger('ModelProcessing.visualization')

def annotating_scores(arg0, arg1, arg2):
    plt.annotate(f'MAPE: {arg0:.2f}', xy=(0.05, 0.85), xycoords='axes fraction')
    plt.annotate(f'NSE: {arg1:.2f}', xy=(0.05, 0.75), xycoords='axes fraction')
    plt.xlabel(arg2)


def get_figures(BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME):
    logger.info(f"Getting figures for {MODEL_NAME}:{NAME}:{VPUID}")

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
                    logger.error(f"Problem with copying the image: {e}")

def plot_domain(domain, fig_output_path):
    """
    Plot a domain array with specific colors:
    - Black for 0 (assuming 0 represents 'active')
    - Green for 1 (assuming 1 represents 'boundary')
    - Red for 2 (assuming 2 represents 'lakes')
    :param domain: A 2D numpy array representing the domain.
    """

    # Check unique values in the domain
    unique_values = np.unique(domain)
    logger.info(f'Unique values in domain: {unique_values}')

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

def plot_streamflow(observed, simulated, title, time_step, name_score, START_YEAR, END_YEAR, nyskip, fig_files_paths, stage, model_log_path, cms_to_cfs=35.3147):
    """Plot the observed and simulated streamflow
    
    Args:
        observed (np.array): Array of observed streamflow values
        simulated (np.array): Array of simulated streamflow values
        title (str): Title for the plot
        time_step (str): Time step for the plot (daily or monthly)
        name_score (float): NSE score to include in the plot
        START_YEAR (int): Start year for the x-axis
        END_YEAR (int): End year for the x-axis
        nyskip (int): Number of years to skip
        fig_files_paths (str): Directory path to save figures
        stage (str): Stage of the model (calibration, verification, etc.)
        model_log_path (str): Path to log file
        cms_to_cfs (float): Conversion factor from cms to cfs
        
    Returns:
        None: Saves the plot to file
    """
    logger.debug(f"Plotting {time_step} streamflow for {title}")
    
    try:
        # Since we don't have actual date indices for the arrays, we need to create the date range
        # and ensure proper alignment
        data_range = pd.date_range(
            start=f'{START_YEAR + nyskip}-01-01', 
            end=f'{END_YEAR}-12-31', 
            freq='D'
        )
        
        # Create DataFrames with dates for proper alignment
        # This assumes that observed and simulated arrays start on the first day of the specified range
        observed_df = pd.DataFrame({'date': data_range[:len(observed)], 'value': observed})
        simulated_df = pd.DataFrame({'date': data_range[:len(simulated)], 'value': simulated})
        
        # Merge based on date to ensure proper alignment
        merged_data = pd.merge(observed_df, simulated_df, on='date', how='inner', suffixes=('_obs', '_sim'))
        
        if merged_data.empty:
            logger.error(f"No matching dates between observed and simulated data for {title}")
            return
            
        logger.info(f"Found {len(merged_data)} matching dates for plotting from {merged_data.date.min()} to {merged_data.date.max()}")
        
        # Calculate performance metrics using the aligned data
        nse_score = nse(merged_data.value_obs.values, merged_data.value_sim.values)
        mape_score = mape(merged_data.value_obs.values, merged_data.value_sim.values)
        pbias_score = pbias(merged_data.value_obs.values, merged_data.value_sim.values)
        
        # Plot the aligned data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(merged_data.date, merged_data.value_obs, label='Observed', color='blue', linewidth=1)
        ax.plot(merged_data.date, merged_data.value_sim, label='Simulated', color='red', linewidth=1)
        ax.set_title(f'{title} {stage} Streamflow')
        ax.set_xlabel('Date')
        ax.set_ylabel('Streamflow (cfs)')
        
        # Set up the x-axis for displaying years as major ticks and months as minor ticks
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(MonthLocator())
        
        # Use grid only for the months (minor ticks)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
        ax.legend()
        
        # Add performance metrics to the plot
        plt.annotate(
            f'NSE: {nse_score:.2f}\nMAPE: {mape_score:.2f}\nPBIAS: {pbias_score:.2f}', 
            xy=(0.05, 0.85), 
            xycoords='axes fraction', 
            fontsize=12
        )
        
        # Ensure directory exists
        output_dir = f'{fig_files_paths}/SF/{stage}/{time_step}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        filename = f'{output_dir}/{name_score:.2f}_{title}_{int(time.time())}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.debug(f"Saved streamflow plot to {filename}")
        
    except Exception as e:
        error_msg = f"Error plotting streamflow for {title}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        with open(model_log_path, 'a') as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} - {error_msg}\n")

def plot_streamflow_monthly(observed_monthly, simulated_monthly, title, time_step, name_score, fig_files_paths, stage, model_log_path):
    """Plot the observed and simulated monthly streamflow
    
    Args:
        observed_monthly (np.array): Array of observed monthly streamflow values with yr and mon columns
        simulated_monthly (np.array): Array of simulated monthly streamflow values with yr and mon columns
        title (str): Title for the plot
        time_step (str): Time step for the plot (should be 'monthly')
        name_score (float): NSE score to include in the plot
        fig_files_paths (str): Directory path to save figures
        stage (str): Stage of the model (calibration, verification, etc.)
        model_log_path (str): Path to log file
        
    Returns:
        None: Saves the plot to file
    """
    logger.debug(f"Plotting monthly streamflow for {title}")
    
    try:
        # For monthly data, we should use proper date alignment
        # Convert to DataFrames if they aren't already
        if not isinstance(observed_monthly, pd.DataFrame):
            logger.warning("Converting observed_monthly to DataFrame for proper date alignment")
            observed_monthly = pd.DataFrame(observed_monthly, columns=['streamflow'])
        
        if not isinstance(simulated_monthly, pd.DataFrame):
            logger.warning("Converting simulated_monthly to DataFrame for proper date alignment")
            simulated_monthly = pd.DataFrame(simulated_monthly, columns=['flo_out'])
        
        # If we have year and month columns in both DataFrames, use them for alignment
        if 'yr' in observed_monthly.columns and 'mon' in observed_monthly.columns and \
           'yr' in simulated_monthly.columns and 'mon' in simulated_monthly.columns:
            
            # Create proper date objects for alignment
            obs_df = observed_monthly.copy()
            sim_df = simulated_monthly.copy()
            
            # Create date strings and convert to datetime
            obs_df['date'] = pd.to_datetime(obs_df['yr'].astype(str) + '-' + 
                                         obs_df['mon'].astype(str).str.zfill(2) + '-01')
            sim_df['date'] = pd.to_datetime(sim_df['yr'].astype(str) + '-' + 
                                         sim_df['mon'].astype(str).str.zfill(2) + '-01')
            
            # Merge the data on date to ensure alignment
            if 'streamflow' in obs_df.columns and 'flo_out' in sim_df.columns:
                merged_data = pd.merge(
                    obs_df[['date', 'streamflow']], 
                    sim_df[['date', 'flo_out']], 
                    on='date', 
                    how='inner'
                )
                
                logger.info(f"Found {len(merged_data)} matching months for plotting from {merged_data.date.min()} to {merged_data.date.max()}")
                
                if merged_data.empty:
                    logger.error(f"No matching dates between observed and simulated data for {title}")
                    return
                
                # Calculate performance metrics
                nse_score = nse(merged_data.streamflow.values, merged_data.flo_out.values)
                mape_score = mape(merged_data.streamflow.values, merged_data.flo_out.values)
                pbias_score = pbias(merged_data.streamflow.values, merged_data.flo_out.values)
                
                # Plot with proper time axis
                plt.figure(figsize=(12, 6))
                plt.grid(linestyle='--', linewidth=0.5)
                
                # Plot the data
                plt.plot(merged_data.date, merged_data.streamflow, label='Observed', color='blue', linewidth=1)
                plt.plot(merged_data.date, merged_data.flo_out, label='Simulated', color='red', linewidth=1)
                
                # Format x-axis to show years and months
                plt.gca().xaxis.set_major_locator(YearLocator())
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
                plt.gca().xaxis.set_minor_locator(MonthLocator())
                plt.gcf().autofmt_xdate()  # Auto-format dates for better readability
                
            else:
                logger.warning("Missing 'streamflow' or 'flo_out' columns, using simple index plotting")
                merged_data = None
        else:
            logger.warning("Missing year/month columns, using simple index plotting")
            merged_data = None
            
        # If we couldn't create merged data with dates, fall back to simple index plotting
        if merged_data is None:
            plt.figure(figsize=(12, 6))
            plt.grid(linestyle='--', linewidth=0.5)
            
            # Plot using index-based alignment
            # Extract the arrays for plotting
            if isinstance(observed_monthly, pd.DataFrame) and 'streamflow' in observed_monthly.columns:
                obs_values = observed_monthly.streamflow.values
            else:
                obs_values = observed_monthly
                
            if isinstance(simulated_monthly, pd.DataFrame) and 'flo_out' in simulated_monthly.columns:
                sim_values = simulated_monthly.flo_out.values
            else:
                sim_values = simulated_monthly
            
            # Make sure lengths match if using index-based alignment
            min_length = min(len(obs_values), len(sim_values))
            obs_values = obs_values[:min_length]
            sim_values = sim_values[:min_length]
            
            # Create month labels for x-axis
            x_values = np.arange(min_length)
            
            # Plot the data
            plt.plot(x_values, obs_values, label='Observed', color='blue', linewidth=1)
            plt.plot(x_values, sim_values, label='Simulated', color='red', linewidth=1)
            
            # Calculate performance metrics
            nse_score = nse(obs_values, sim_values)
            mape_score = mape(obs_values, sim_values)
            pbias_score = pbias(obs_values, sim_values)
            
        # Set labels and title (common for both plotting methods)
        plt.title(f'{title} {stage} Monthly Streamflow')
        plt.xlabel('Date' if merged_data is not None else 'Month')
        plt.ylabel('Streamflow (cfs)')
        plt.legend()
        
        # Add performance metrics to the plot
        plt.annotate(
            f'NSE: {nse_score:.2f}\nMAPE: {mape_score:.2f}\nPBIAS: {pbias_score:.2f}', 
            xy=(0.05, 0.85), 
            xycoords='axes fraction', 
            fontsize=12
        )
        
        # Ensure directory exists
        output_dir = f'{fig_files_paths}/SF/{stage}/{time_step}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        filename = f'{output_dir}/{name_score:.2f}_{title}_{int(time.time())}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.debug(f"Saved monthly streamflow plot to {filename}")
        
    except Exception as e:
        error_msg = f"Error plotting monthly streamflow for {title}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        with open(model_log_path, 'a') as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} - {error_msg}\n")

def plot_global_best_improvement(global_best_scores, username, VPUID, LEVEL, NAME, MODEL_NAME):
    """Plot the improvement of global best score over iterations
    
    Args:
        global_best_scores (list): List of global best scores for each iteration
        username (str): Username for path construction
        VPUID (str): VPUID identifier
        LEVEL (str): Level identifier (e.g., 'huc12')
        NAME (str): Name identifier
        MODEL_NAME (str): Model name
        
    Returns:
        None: Saves the plot to file
    """
    logger.debug(f"Plotting global best improvement for {MODEL_NAME}:{NAME}:{VPUID}")
    
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(global_best_scores, color='b', marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Value')
        plt.title('Global Best Improvement')
        plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
        
        # Make sure the directory exists
        output_dir = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/figures_{MODEL_NAME}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        filename = f"{output_dir}/GlobalBestImprovement.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        
        logger.debug(f"Saved global best improvement plot to {filename}")
        
    except Exception as e:
        logger.error(f"Error plotting global best improvement: {str(e)}", exc_info=True)
