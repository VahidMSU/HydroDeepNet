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


def creating_plots(simulated, observed, title, SCENARIO, fig_files_paths, BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME, stage ):

    for time_step in ["Daily","Monthly"]:
        fig = plt.figure(figsize = [8, 4])
        simulated_temp = simulated.copy()
        observed_temp = observed.copy()
        simulated_temp ['date'] = observed_temp['date'].values
        simulated_temp.set_index('date', inplace=True)
        observed_temp.set_index('date', inplace=True)
        total_length=len(observed_temp)
        simulated_temp = simulated_temp[observed_temp.streamflow>0]
        observed_temp = observed_temp[observed_temp.streamflow>0]

        if time_step=="Monthly":

            simulated_monthly = simulated_temp.groupby(pd.Grouper(freq='ME'))['flo_out'].sum()
            observed_monthly  = observed_temp.groupby(pd.Grouper(freq='ME'))['streamflow'].sum()
            dates = observed_monthly.index
            plt.plot(dates, simulated_monthly.values, color='blue', linewidth=0.3)
            plt.plot(dates, observed_monthly.values, color='red', linewidth=0.5)
            monthly_mape_value = mape(observed_monthly, simulated_monthly)
            monthly_nse_value = nse(observed_monthly, simulated_monthly)
            pbias_value = pbias(observed_monthly, simulated_monthly)
            annotating_scores(
                monthly_mape_value, monthly_nse_value, 'Monthly'
            )
            plt.title(title)
            plt.ylabel('Average Flow (cfs)')
            plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #if len(simulated)>10:
            #    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=24))
            #else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            plt.legend(['Simulated', 'Observed'])
            plt.tight_layout()
            os.makedirs(os.path.join(fig_files_paths, f"figures_{MODEL_NAME}_{stage}_monthly"), exist_ok=True)

            plt.savefig(os.path.join(fig_files_paths, f"figures_{MODEL_NAME}_{stage}_monthly/{round(float(monthly_nse_value), 2)}_{title}_{SCENARIO}.jpeg"), dpi=200)

            writing_performance_scores (BASE_PATH, MODEL_NAME, LEVEL, VPUID,NAME,title, time_step, monthly_nse_value, monthly_mape_value, pbias_value, SCENARIO, stage)

        if time_step=="Daily":

            dates = observed_temp.index
            plt.plot(dates, simulated_temp.flo_out.values, color='blue', linewidth=0.3)
            plt.plot(dates, observed_temp.streamflow.values, color='red', linewidth=0.5)
            daily_mape_value = mape(observed_temp.streamflow.values, simulated_temp.flo_out.values)
            daily_nse_value = nse(observed_temp.streamflow.values, simulated_temp.flo_out.values)
            pbias_value = pbias(observed_temp.streamflow.values, simulated_temp.flo_out.values)
            annotating_scores(daily_mape_value, daily_nse_value, 'Daily')
            plt.title(f'{title}_{MODEL_NAME}')
            plt.ylabel('Flow (cfs)')
            plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            #if len(simulated)>10:
            #    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=24))
            #else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()
            plt.legend(['Simulated', 'Observed'])
            plt.tight_layout()
            os.makedirs(os.path.join(fig_files_paths, f"figures_{MODEL_NAME}_{stage}_daily"), exist_ok=True)
            plt.savefig(os.path.join(fig_files_paths, f"figures_{MODEL_NAME}_{stage}_daily/{round(float(daily_nse_value), 2)}_{title}_{SCENARIO}.jpeg"), dpi=200)


            writing_performance_scores (BASE_PATH, MODEL_NAME, LEVEL, VPUID, NAME,title, time_step, daily_nse_value, daily_mape_value, pbias_value, SCENARIO, stage)
        if stage == 'calibration':
            get_figures(BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME)

    return -1*(monthly_nse_value+daily_nse_value)


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


def writing_performance_scores (BASE_PATH, MODEL_NAME, LEVEL, VPUID, NAME,title , time_step, NSE, MPE, PBIAS, SCENARIO='RandomName', stage='RandomName'):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    performance_scores_path=os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{stage}_performance_scores.txt')
    if not os.path.exists(performance_scores_path):
        with open(performance_scores_path, 'w') as file:
            file.write(f'model performance scores. First creation: {current_time}\n')
            file.write(f'Time_of_writing\tstation\ttime_step\tMODEL_NAME\tSCENARIO\tNSE\tMPE\tPBIAS\n')
    else:
        with open(performance_scores_path, 'a') as file:
            file.write(f'{current_time}\t{title}\t{time_step}\t{MODEL_NAME}\t{SCENARIO}\t{NSE}\t{MPE}\t{PBIAS}\n')
            
    if stage == 'sensitivity': 
         os.remove(os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Scenario_{SCENARIO}'))