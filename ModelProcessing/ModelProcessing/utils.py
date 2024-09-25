import contextlib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import glob
import pandas as pd
import numpy as np
import shutil
import psutil
import time
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

### add date in logging



def check_number_of_files(output_dir):
		if not os.path.exists(output_dir):
				return 0, 0
		pcp_files = [f for f in os.listdir(output_dir) if '.pcp' in f]
		tmp_files = [f for f in os.listdir(output_dir) if '.tmp' in f]
		return len(pcp_files), len(tmp_files)


def write_model_parameters(param_files, params, problem, operation_types, scenario_TxtInOut, original_TxtInOut):
		logging.info(f"Writing model parameters to {scenario_TxtInOut}")
		params_dict = dict(zip(problem['names'], params))
		for file_name, variables in param_files.items():
				df = read_swat_input_data(original_TxtInOut, file_name=file_name)
				for var in variables:
						operation = operation_types.get(var, 'replace')  # default to 'replace' if not found
						value = params_dict[var]
						if 'hhc' in var or 'sy' in var or 'k_sb' in var or 'thickness_sb' in var:
								zone_value  = int(var.split('_')[0])-1
								column_name = var.split('_')[1:]
								if operation == 'add':
										df.loc[zone_value,column_name] += value
								elif operation == 'multiply':
										df.loc[zone_value,column_name] *= value
								elif operation == 'percentage':
										df.loc[zone_value,column_name] += df[zone_value,column_name] * (value / 100)
								elif operation == 'replace':
										df.loc[zone_value,column_name] = value
						else:
								df[var] = df[var].astype(float)
								if operation == 'replace':
										df[var] = value
								elif operation == 'multiply':
										df[var] *= value
								elif operation == 'add':
										df[var] += value
								elif operation == 'percentage':
										df[var] += df[var] * (value / 100)

				write_swat_input_data(scenario_TxtInOut, df, file_name=file_name)


def log_errors(filename, line_to_append):
		current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
		if not os.path.exists(filename):
				logging.info(f"Creating log file: {filename}")
				os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as file:
				file.write(f"{current_time} - {line_to_append}\n")

def load_initial_data(initial_points_path, initial_values_path):
		if os.path.exists(initial_points_path) and os.path.exists(initial_values_path):
				initial_points = np.loadtxt(initial_points_path, delimiter=',')
				initial_objective_values = np.loadtxt(initial_values_path, delimiter=',')
				return initial_points, initial_objective_values
		return None, None
def filling_observations(df_complete, observed):
		"""
		Fill missing observations in a time series using the Holt-Winters Exponential Smoothing model.

		This function takes a complete DataFrame with dates and a second DataFrame with observed streamflow values.
		It merges these two DataFrames, then applies the Holt-Winters Exponential Smoothing model to interpolate
		or extrapolate the streamflow values for dates where observations are missing.

		Parameters:
		- df_complete (pd.DataFrame): A DataFrame containing a complete set of dates.
		It must have a column named 'date'.
		- observed (pd.DataFrame): A DataFrame containing observed streamflow values.
		This DataFrame should have a 'date' column and a 'streamflow' column.

		Returns:
		- pd.DataFrame: A DataFrame similar to df_complete but with the 'streamflow' column filled
										with both the original observed values and the interpolated/extrapolated values
										for missing dates, based on the Holt-Winters model.
		"""

		observed_with_gaps = pd.merge(df_complete, observed, on='date', how='left')
		model = ExponentialSmoothing(observed['streamflow'], seasonal='add', seasonal_periods=12)
		results = model.fit()

		# Update 'streamflow' in observed_with_gaps with predictions
		observed_with_gaps['streamflow'] = results.predict(start=observed_with_gaps.index[0], end=observed_with_gaps.index[-1])

		return observed_with_gaps


def checking_the_number_true_stations(BASE_PATH, LEVEL, VPUID, NAME, END_YEAR, START_YEAR, nyskip):

		"""checking the number of stations based on a defined period and percentage of missing observations (less than 10% is only accepted)
		"""


		#stations_path = glob.glob(os.path.join(BASE_PATH, fr"SWATplus_by_VPUID/{LEVEL}/{NAME}/streamflow_data/",'*.csv'))
		stations_path = glob.glob(fr"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/streamflow_data/*.csv")


		number_of_stations = 0
		for station_path in stations_path:
				obs = pd.read_csv(station_path, index_col='Unnamed: 0', parse_dates=['date'])
				date_range = pd.date_range(start=f'{START_YEAR+nyskip}-01-01', end=f'{END_YEAR}-12-31', freq='D')
				df_complete = pd.DataFrame(date_range, columns=['date'])
				missing_dates = df_complete[~df_complete['date'].isin(obs['date'])]
				gap_length = len(missing_dates)
				total_length = len(df_complete)
				gap_percent = gap_length/total_length
				if gap_percent<0.10:
						number_of_stations = number_of_stations+1

		return number_of_stations

def ram_usage():
		# Get memory details
		memory = psutil.virtual_memory()
		# Total memory
		total_memory = memory.total / (1024 ** 3)  # Convert bytes to GB
		logging.info(f"Total Memory: {total_memory:.2f} GB")
		# Used memory
		used_memory = memory.used / (1024 ** 3)  # Convert bytes to GB
		logging.info(f"Used Memory: {used_memory:.2f} GB")
		# Memory usage percentage
		memory_usage_percent = memory.percent
		logging.info(f"Memory Usage: {memory_usage_percent}%")

		return used_memory

def count_processes(process_name):
		"""
		Counts the number of processes with a name that includes the given process_name.

		Args:
		- process_name (str): The name or partial name of the process to count.

		Returns:
		- int: The count of processes matching the given name.
		"""
		process_count = 0
		for process in psutil.process_iter(['name']):
				with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
						if process_name.lower() in process.info['name'].lower():
								process_count += 1
		return process_count

def is_cpu_usage_low(intervals=2, n_processes=150, swat_model=True, process_name='swatplus'):

		"""Check if the current CPU usage is lower than the threshold."""
		n_processes=150
		CPU_USAGE_AVERAGE = psutil.cpu_percent(interval = intervals)
		logging.info(f"CPU Usage: {CPU_USAGE_AVERAGE}%")
		used_memory = ram_usage()
		logging.info(f"RAM Usage: {used_memory:.2f} GB")
		if swat_model:
				swatplus_process_count = count_processes(process_name = 'swatplus')
				logging.info(f"Number of SWAT+ processes: {swatplus_process_count}")
				#time.sleep(15)
				return swatplus_process_count<n_processes and CPU_USAGE_AVERAGE < 90 and used_memory<500
		else:
				process_count = count_processes(process_name = process_name)

				return process_count<n_processes and CPU_USAGE_AVERAGE < 90 and used_memory<500
		################################# statistical functions ########################################################################################


##############################################  Clean-up functions    #######################################################################

def removing_directory(path):
		if os.path.exists(path):
				try:
						shutil.rmtree(path)
				except Exception as e:
						logging.info(f"Error removing folder: {e}")
def copy_files(NAME, VPUID, MODEL_NAME, path_to_copy, OUTPUT_PATH):

	logging.info(f"copy_files: {VPUID}, {NAME}, {MODEL_NAME}")
	try:

		if not os.path.exists(os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME)):
			os.makedirs(os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME), exist_ok=True)
			logging.info(f'created path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME)}')

		if not os.path.exists(os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME, MODEL_NAME, 'streamflow_data')):
			if not os.path.exists(path_to_copy['streamflow']):
				return True

			shutil.copytree(path_to_copy['streamflow'], os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME, 'streamflow_data'), dirs_exist_ok=True)
			logging.info(f'creating path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME, MODEL_NAME, "streamflow_data")}')

		if not os.path.exists(os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME, MODEL_NAME)):
			if not os.path.exists(path_to_copy['model']):
				return True
			shutil.copytree(path_to_copy['model'], os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME, MODEL_NAME))
			logging.info(f'created path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME, MODEL_NAME)}')

		if 'cal' in path_to_copy and path_to_copy['cal']:
			shutil.copy2(f"/data/MyDataBase/bin/{os.path.basename(path_to_copy['cal'])}", os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME))
			logging.info(f'created path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME)}')


		if 'best' in path_to_copy and path_to_copy['best'] and 'local' in path_to_copy and path_to_copy['local']:
			try:
				shutil.copy2(path_to_copy['best'], os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME))
				shutil.copy2(path_to_copy['local'], os.path.join(OUTPUT_PATH, VPUID, 'huc12', NAME))
			except FileNotFoundError as e:
				logging.info(f'File not found: {e}')
		logging.info(f'target path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME, MODEL_NAME)}')

	except PermissionError as e:
		logging.info(f'Permission denied: {e}')
	logging.info(f'path: {os.path.join(OUTPUT_PATH, VPUID, "huc12", NAME, MODEL_NAME)}')
	return False



def delete_previous_runs(directory_path):
		if not os.path.exists(directory_path):
				return
		folders_in_directory = os.listdir(directory_path)
		for folder_name in folders_in_directory:
				try:
						if folder_name not in ['Default'] or 'verification' in folder_name:
								folder_path = os.path.join(directory_path, folder_name)
								if os.path.isdir(folder_path):
										try:
												shutil.rmtree(folder_path)
										except Exception as e:
												logging.info(f'Not deleted: {folder_path}, due to error: {e}')

				except Exception as e:
						logging.info(f'Not deleted: {folder_path}, due to error: {e}')


def delete_previous_figures(directory_path):
		"""
		Delete all figures in the specified directory.

		Parameters:
		- directory_path (str): the path to the directory containing the figures to be deleted

		Returns:
		- None
		"""

		# Construct a list of all figures in the directory
		file_types = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.bmp"]
		for file_type in file_types:
				for file_path in glob.glob(os.path.join(directory_path, file_type)):
						try:
								os.remove(file_path)
						except Exception as e:
								logging.info(f"Error deleting {file_path}: {e}")



def clean_up(SCV_args):

		"""
		Clean up files related to the calibration and sensitivity analysis of a model.

		Parameters:
		BASE_PATH (str): Base directory path.
		LEVEL (str): Level of the model (e.g., 'huc12').
		NAME (str): Name of the model.
		MODEL_NAME (str): Name of the model file.
		bayesian_model_path (str): Path to the Bayesian model file.
		sensitivity_flag (bool): Flag to delete sensitivity analysis files.
		calibration_flag (bool): Flag to delete calibration files.
		"""
		BASE_PATH = SCV_args['BASE_PATH']
		LEVEL = SCV_args['LEVEL']
		VPUID = SCV_args['VPUID']
		NAME = SCV_args['NAME']
		MODEL_NAME = SCV_args['MODEL_NAME']
		sensitivity_flag = SCV_args['sensitivity_flag']
		calibration_flag = SCV_args['calibration_flag']

		cal_file_path = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/')
		scenarios_path = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios')

		# Delete scenarios and figures
		delete_previous_runs(scenarios_path)

		if calibration_flag == True:
				_extracted_from_clean_up_30(cal_file_path, MODEL_NAME, BASE_PATH)
		if sensitivity_flag == True:
				delete_previous_figures(os.path.join(cal_file_path, f"figures_{MODEL_NAME}_sensitivity_daily"))
				delete_previous_figures(os.path.join(cal_file_path, f"figures_{MODEL_NAME}_sensitivity_monthly"))

				sensitivity_files = ['initial_points', 'morris_Si', 'initial_values']
				for file_name in sensitivity_files:
						file_path = os.path.join(cal_file_path, f'{file_name}_{MODEL_NAME}.csv')
						try:
								os.remove(file_path)
						except FileNotFoundError:
								logging.info(f'File not found: {file_path}')
						except Exception as e:
								logging.info(f'Error removing {file_name}: {e}')

		logging.info(f'clean up is done.{MODEL_NAME}:{NAME}:{VPUID}. calibration_flag:{calibration_flag}. sensitivity_flag:{sensitivity_flag}\t')

# TODO Rename this here and in `clean_up`
def _extracted_from_clean_up_30(cal_file_path, MODEL_NAME, BASE_PATH):
		delete_previous_figures(os.path.join(cal_file_path, f"figures_{MODEL_NAME}_calibration_daily"))
		delete_previous_figures(os.path.join(cal_file_path, f"figures_{MODEL_NAME}_calibration_monthly"))
		delete_previous_figures(os.path.join(cal_file_path, f"calibration_figures_{MODEL_NAME}"))

		# Delete calibration files if requested
		cal_file_path_target = os.path.join(cal_file_path, f'cal_parms_{MODEL_NAME}.cal')

		if os.path.exists(cal_file_path_target):
				os.remove(cal_file_path_target)
		shutil.copy2(os.path.join(BASE_PATH, f'bin/cal_parms_{MODEL_NAME}.cal'), cal_file_path)


################################################  SWAT input manipulation functions ######################################################

def update_swat_codes_bsn(TxtInOut_path, pet=0, cn=2):
		codes_bsn = read_swat_input_data(TxtInOut_path, 'codes.bsn')
		codes_bsn['pet'] = pet
		codes_bsn['cn'] = cn
		write_swat_input_data(TxtInOut_path, codes_bsn, 'codes.bsn')

def nyskip_define(TxtInOut_path, nyskip=1):
		with open(os.path.join(TxtInOut_path, "print.prt")) as file:
				lines=file.readlines()
				lines[2] = f"{int(nyskip)}           0         0         0         0         0\n"
		with open(os.path.join(TxtInOut_path, "print.prt"), 'w') as file:
				file.writelines(lines)

def activate_hru_print(TxtInOut_path):
		with open(os.path.join(TxtInOut_path,'print.prt'), 'r') as file:
				lines=file.readlines()
				for i in range(len(lines)):
						if 'hru_wb' in lines[i]:
								lines[i]='hru_wb                       n             y             y             y  \n'

		with open(os.path.join(TxtInOut_path,'print.prt'), 'w') as file:
				file.writelines(lines)
import pandas as pd
import os
import pandas as pd
import os

def read_swat_input_data(TxtInOut_path, file_name):
		# sourcery skip: extract-method
		if file_name == 'gwflow.input':
				dataframe = pd.DataFrame(columns=['zone', 'hhc', 'sy', 'k_sb', 'thickness_sb'])
				with open(os.path.join(TxtInOut_path, file_name)) as file:
						lines = file.readlines()
						datas = ['Aquifer Hydraulic', 'Aquifer Specific Yield', 'Streambed Hydraulic', 'Streambed Thickness']
						names = ['hhc', 'sy', 'k_sb', 'thickness_sb']
						for data, name in zip(datas, names):
								for i, line in enumerate(lines):
										if data in line:
												number_of_zone = int(lines[i + 1].strip())
												for j in range(number_of_zone):
														zone, value = lines[i + 2 + j].split()
														zone = int(zone)
														value = float(value)
														# Check if zone is already in dataframe
														if zone in dataframe['zone'].values:
																dataframe.loc[dataframe['zone'] == zone, name] = value
														else:
																# Create a new row and use concat
																new_row = pd.DataFrame({'zone': [zone], name: [value]})
																new_row[name] = new_row[name].astype(float)
																if dataframe.empty:
																		dataframe = new_row.copy()
																else:
																	dataframe = pd.concat([dataframe, new_row], ignore_index=True, sort=False).copy()
				return dataframe
		else:
				with open(os.path.join(TxtInOut_path, file_name)) as file:
						lines = file.readlines()
						data = []
						start_row = 2
						headers = lines[1].split()
						# Check if the last header is "description" and exclude it
						if headers[-1].lower() == 'description':
								headers = headers[:-1]
						for line in lines[start_row:]:
								row_data = line.split()
								if len(row_data) > len(headers):
										row_data = row_data[:len(headers)]
								data.append(row_data)
				return pd.DataFrame(data, columns=headers)


def write_swat_input_data(TxtInOut_path, df, file_name):

		if file_name=='gwflow.input':
				write_swatgwflow_input_data(TxtInOut_path, file_name, df)
		else:
				with open(os.path.join(TxtInOut_path ,file_name), 'w') as file:
						current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
						readme_line = f"{file_name}: by myself on {current_date}"
						# Write the readme line
						file.write(readme_line + '\n')
						# Write the headers
						headers = "  ".join(df.columns)
						file.write(f"{headers}\n")
						# Write the rows
						for _, row in df.iterrows():
								row_str = "  ".join(map(str, row.values))
								file.write(f"{row_str}\n")


def write_swatgwflow_input_data(TxtInOut_path, file_name, df):
		input_path = os.path.join(TxtInOut_path, file_name)
		datas = ['Aquifer Hydraulic', 'Aquifer Specific Yield', 'Streambed Hydraulic', 'Streambed Thickness']
		names = ['hhc', 'sy', 'k_sb', 'thickness_sb']

		# Read the entire file into memory
		with open(input_path, 'r') as file:
				lines = file.readlines()

		# Update the lines with new data
		for data, name in zip(datas, names):
				for i, line in enumerate(lines):
						if data in line:
								number_of_zone = int(lines[i+1].strip())
								for j in range(number_of_zone):
										zone = int(lines[i+2+j].split()[0])
										if zone in df['zone'].values:
												# Update the line with the new value
												value = df.loc[df['zone'] == zone, name].values[0]
												lines[i+2+j] = f"{zone} {value}\n"

		# Write the updated lines back to the file
		with open(input_path, 'w') as file:
				file.writelines(lines)


def update_time(TxtInOut_path, START_YEAR, END_YEAR):
		time = read_swat_input_data(TxtInOut_path, file_name='time.sim')
		time['yrc_start']=START_YEAR
		time['yrc_end']=END_YEAR
		write_swat_input_data(TxtInOut_path, time, file_name='time.sim')

def read_control_file(cal_parms):
		param_files = {}
		operation_types = {}
		problem = {'num_vars': 0, 'names': [], 'bounds': []}

		for _, row in cal_parms.iterrows():
				name = row['name']
				file_name = row['file_name']
				min_val = float(row['min'])  # Explicitly cast to float
				max_val = float(row['max'])
				operation = row['operation']

				if file_name not in param_files:
						param_files[file_name] = []
				param_files[file_name].append(name)

				operation_types[name] = operation
				problem['names'].append(name)
				problem['bounds'].append((min_val, max_val))
		problem['num_vars'] = len(problem['names'])
		return param_files, operation_types, problem


############################################  Writing output functions  #########################################################
def update_print_prt_file(TxtInOut_path, daily_flow_printing=True, hru_printing=False):
		# Read the existing lines
		file_path = os.path.join(TxtInOut_path, 'print.prt')

		with open(file_path, 'r') as file:
				lines = file.readlines()
				for i, line in enumerate(lines):
						if 'objects' in line:
								start_replacement_idx=i+1

		# Process the DataFrame
		print_prt = pd.read_csv(file_path, skiprows=9, sep='\s+')
		print_prt.iloc[:, 1:] = 'n'

		if daily_flow_printing==True:
				print_prt.loc[print_prt[print_prt.objects == 'channel_sd'].index, 'monthly'] = 'y'
				print_prt.loc[print_prt[print_prt.objects == 'channel_sd'].index, 'daily'] = 'y'
		if hru_printing==True:
				print_prt.loc[print_prt[print_prt.objects == 'hru_wb'].index, 'monthly'] = 'y'
				print_prt.loc[print_prt[print_prt.objects == 'hru_wb'].index, 'yearly'] = 'y'
				print_prt.loc[print_prt[print_prt.objects == 'hru_wb'].index, 'avann'] = 'y'

		# Format the DataFrame rows with 4 spaces between columns
		formatted_lines = ['    '.join(f"{item}" for item in row) for row in print_prt.values]
		end_replacement_idx = start_replacement_idx + len(formatted_lines)
		# Replace the lines in the original list of lines with the formatted lines
		lines[start_replacement_idx:end_replacement_idx] = [line + '\n' for line in formatted_lines]
		# Write the lines back to the file
		with open(file_path, 'w') as file:
				file.writelines(lines[:start_replacement_idx])  # Original header
				file.writelines(lines[start_replacement_idx:end_replacement_idx])  # Formatted lines
				if end_replacement_idx < len(lines):
						file.writelines(lines[end_replacement_idx:])  # The rest of the original content

def checking_models_under_processing(process_name, LEVEL):
		""" this function checks the number of models under processing and returns the names of the models under processing"""
		count = 0
		NAME_under_processing=[]

		for process in psutil.process_iter(['pid', 'name', 'exe']):

				try:
						if process_name in process.info['name'].lower() and "editor" not in process.info['name'].lower() and LEVEL in process.info['exe']:
								count += 1

								exe_path = process.info['exe'].split(f'{LEVEL}')[1]
								first_part_of_path = exe_path.split('//')[1]
								NAME_under_processing.append(str(first_part_of_path))
				except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
						# Handle processes that have terminated or cannot be accessed
						continue

		logging.info(f"Total number of processes named '{process_name}': {count}")	
		return(list(np.unique(NAME_under_processing)))