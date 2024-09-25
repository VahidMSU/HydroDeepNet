import h5py
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
from dateutil.relativedelta import relativedelta
from unit_global import list_of_variables

def clean_dir(fig_path):
	if os.path.exists(fig_path):
		shutil.rmtree(fig_path)
	os.makedirs(fig_path)
	return fig_path
	
def plot_time_series(model_var, monthly_index, var_name, unit, fig_path,time_step,dataset_name):
	""" Plot time series of the SWAT+ output h5 file"""
	et_975 = np.percentile(model_var, 97.5, axis=1)
	et_025 = np.percentile(model_var, 2.5, axis=1)
	et_median = np.median(model_var, axis=1)
	
	fig, ax = plt.subplots(figsize=(10, 6))  
	ax.fill_between(monthly_index, et_975, et_025, alpha=0.5, color = 'green', label='95th')
	ax.plot(monthly_index, et_median, label='Median')
	
	ax.legend()
	# trun on minor ticks for x-axis but no label should be attached to them
	ax.set_xticks(monthly_index, minor=True)
	ax.set_ylabel(f'({unit})')
	ax.grid(True, which="both", ls="--", alpha=0.25)
	ax.set_title(f"{dataset_name.split('_')[0].capitalize()}s temporal variablity for: {var_name} ({time_step})")
	
	plt.savefig(os.path.join(fig_path, f'{var_name}_{time_step}.png'), dpi = 300)
	plt.close()
def define_time_index(time_step, years, months):
	""" Define time index for the SWAT+ output h5 file"""
	
	if time_step == 'day':
		frequency = 'D'
	elif time_step == 'mon':
		frequency = 'ME'
	elif time_step == 'yr':
		frequency = 'YE'
		
	start_date = f'{years[0]}-{months[0]:02d}'
	end_date = pd.to_datetime(f'{years[-1]}-{months[-1]:02d}') + relativedelta(months=1)
	return pd.date_range(start=start_date, end=end_date, freq= frequency)

def visualize_swat_h5(path, fig_path, var_name, dataset_name, time_step):
	""" Visualize SWAT+ output h5 file"""	
	
	with h5py.File(path, 'r') as f:
		try:
			dataset = f[f'{dataset_name}/{time_step}/']   # shape: (time, hru)
			print(f'#################### {dataset_name} ### {time_step} ####################')
		except KeyError:
			print(f'No {dataset_name}/{time_step} dataset in the h5 file')
			return
		model_var = dataset[var_name][:]
		unit = f[f'{dataset_name}/mon/{var_name}'].attrs['unit']
		months = f['metadata/months']
		years = f['metadata/years']
		
		monthly_index = define_time_index(time_step, years, months)

		plot_time_series(model_var, monthly_index, var_name, unit, fig_path, time_step, dataset_name)
import os
from concurrent.futures import ThreadPoolExecutor
from unit_global import list_of_variables

def visualize_swat_hrus_channels_variables():
    """ Visualize SWAT+ output h5 file using parallel processing """
    DIC = "D:/MyDataBase"
    path = os.path.join(DIC, 'codes/SWAT_H5/', "SWAT_OUTPUT.h5")
    fig_path = "/data/MyDataBase/SWATGenXAppData/codes/swat_h5/figs"
    clean_dir(fig_path)

    dataset_names = ['hru_wb', 'channel_sd']
    tasks = []

    with ThreadPoolExecutor() as executor:
        for dataset_name in dataset_names:
            variables = list_of_variables(dataset_name)
            for time_step in ['yr', 'mon']:
                for var_name in variables:
                    tasks.append(executor.submit(visualize_swat_h5, path, fig_path, var_name, dataset_name, time_step))

    # Wait for all tasks to complete
    for task in tasks:
        task.result()  # This will raise exceptions if any occurred during the thread execution

if __name__ == "__main__":
    visualize_swat_hrus_channels_variables()

