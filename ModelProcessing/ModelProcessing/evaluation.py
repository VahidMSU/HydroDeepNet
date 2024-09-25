import os
import shutil
import subprocess
import logging
import pandas as pd
import time
import glob
import logging
from uuid import uuid4 as random_scenario_name_generator
try:
	from ModelProcessing.utils import log_errors, write_model_parameters
	from ModelProcessing.utils import filling_observations
	from ModelProcessing.visualization import creating_plots
except Exception:
	from utils import log_errors, filling_observations, write_model_parameters
	from visualization import creating_plots
### add time to loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def simulate_and_evaluate_swat_model_wrapper(params, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, problem, param_files, operation_types, TxtInOut, SCENARIO):
		
		evaluator = SwatModelEvaluator(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage)
		
		
		return evaluator.simulate_and_evaluate_swat_model(params, problem, param_files, operation_types, TxtInOut, SCENARIO)


class SwatModelEvaluator:
		def __init__(self, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage):
				self.BASE_PATH = BASE_PATH
				self.VPUID = VPUID
				self.LEVEL = LEVEL
				self.NAME = NAME
				self.MODEL_NAME = MODEL_NAME
				self.START_YEAR = START_YEAR
				self.END_YEAR = END_YEAR
				self.nyskip = nyskip
				self.no_value = no_value
				self.stage = stage
				self.model_log_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{str(NAME)}/log.txt")
				self.cms_to_cfs = 35.3147
		def prepare_scenario_files(self, original_TxtInOut, scenario_TxtInOut, execution_file):

				shutil.copytree(original_TxtInOut, scenario_TxtInOut, dirs_exist_ok=True)
				shutil.copy(execution_file, scenario_TxtInOut)

		def define_timeout(self, hru_new_target):
				list_of_hrus = pd.read_csv(hru_new_target, skiprows=1)
				number_of_hrus = len(list_of_hrus)
				if number_of_hrus > 10000:
						return (
								max(2 * number_of_hrus, 18 * 3600)
								if number_of_hrus > 30000
								else max(number_of_hrus, 12 * 3600)
						)
				else:
						return max(number_of_hrus, 8 * 3600)

		def define_timeout_new(self, hru_new_target):
				# Read the list of HRUs
				list_of_hrus = pd.read_csv(hru_new_target, skiprows=1)
				number_of_hrus = len(list_of_hrus)

				# Define the regression parameters
				slope = 1.56e-3
				intercept = 2.49

				# Calculate the execution time in minutes using the regression equation
				execution_time_minutes = slope * number_of_hrus + intercept

				# Convert execution time to seconds
				execution_time_seconds = execution_time_minutes * 60
				if self.stage == 'calibration':
						return max(execution_time_seconds, 3 * 3600)
				else:
						return max(execution_time_seconds, 8 * 3600)



		def simulate_and_evaluate_swat_model(self, params, problem, param_files, operation_types, original_TxtInOut, SCENARIO = None):
				if SCENARIO is None:
						SCENARIO = str(random_scenario_name_generator())
				else:
						SCENARIO = SCENARIO
				scenario_TxtInOut = os.path.join(
						self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Scenario_{SCENARIO}/'
				)
				hru_new_target = os.path.join(
						self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/hru.con'
				)
				model_log_path       = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/log.txt")
				general_log_path     = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/log.txt')
				execution_file       = os.path.join(self.BASE_PATH, 'bin/swatplus')
				fig_files_paths      = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}')
				streamflow_data_path = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/streamflow_data/")

				self.prepare_scenario_files(original_TxtInOut, scenario_TxtInOut, execution_file)

				logging.info(f'writing model parameters for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}\n')


				write_model_parameters(param_files, params, problem, operation_types,
														scenario_TxtInOut, original_TxtInOut)


				start_model_time = time.time()
				timeout_threshold = self.define_timeout_new(hru_new_target)

				try:
						subprocess.run(["/data/MyDataBase/bin/swatplus"],
                     cwd=scenario_TxtInOut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     timeout=timeout_threshold)
						end_model_time = time.time()
						duration = end_model_time - start_model_time
						duration = round(duration / (60 * 60), 2)
						message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} execution terminated within {duration} hours\n'
				except subprocess.TimeoutExpired:
						message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} wall-time met: {round(timeout_threshold / (60 * 60), 2)} hours\n'

				log_errors(model_log_path, message)
				log_errors(general_log_path, message)
				logging.info(message)
				return self.model_evaluation(scenario_TxtInOut, streamflow_data_path, fig_files_paths, SCENARIO)
		def check_empty_file(self, file):
				return bool(os.path.exists(file) and os.path.getsize(file) != 0)
		def model_evaluation(self, scenario_TxtInOut, streamflow_data_path, fig_files_paths, SCENARIO):
				os.chdir(scenario_TxtInOut)
				file_path = "channel_sd_day.txt"
				if os.path.exists(file_path) and os.path.getsize(file_path) != 0:
						usecols = ['day', 'mon', 'yr', 'gis_id', 'flo_out']
						chunks = list(
								pd.read_csv(
										file_path,
										skiprows=[0, 2],
										usecols=usecols,
										chunksize=10 ** 6,
										sep=' ',
										skipinitialspace=True,
								)
						)
						channel_sd_day_data = pd.concat(chunks, axis=0)

						files = glob.glob(f'{streamflow_data_path}*.csv')
						for file in files:
							### remove files name that their size is 0
							if not self.check_empty_file(file):
								files.remove(file)

						if channel_sd_day_data.empty:
								message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} SWAT simulation output is empty, returning {self.no_value}\n'
								log_errors(self.model_log_path, message)
								logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} SWAT simulation output is empty, returning {self.no_value}')
								return self.no_value

						else:
								channel_sd_day_data['date'] = pd.to_datetime(
										channel_sd_day_data['yr'].astype(str) + '-' + channel_sd_day_data['day'].astype(str), format='%Y-%j'
								)
								objective_function_values = 0
								for file in files:

										objective_function_values = self.get_objective_value(
												file, channel_sd_day_data, self.START_YEAR, self.END_YEAR, self.nyskip, fig_files_paths, SCENARIO
										) + objective_function_values
										logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} objective_function_values: {objective_function_values}')

								return objective_function_values
				else:
						message = f'No SWAT simulation file {file_path} exists, returning {self.no_value}\n'
						log_errors(self.model_log_path, message)
						logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No SWAT simulation file {file_path} exists, returning {self.no_value}')
						return self.no_value

		def get_objective_value(self, file, channel_sd_day_data, START_YEAR, END_YEAR, nyskip, fig_files_paths, SCENARIO):
				logging.info(f'start year: {START_YEAR + nyskip}\tend year: {END_YEAR}   ')

				obs = pd.read_csv(file, parse_dates=['date'])
				file = os.path.basename(file)
				observed = obs[(obs.date >= f'{START_YEAR + nyskip}-01-01') & (obs.date <= f'{END_YEAR}-12-31')].reset_index(drop=True)
				logging.info(f'length of observed data: {len(observed)}\tlength of observed data: {len(observed)}')
				date_range = pd.date_range(start=f'{START_YEAR + nyskip}-01-01', end=f'{END_YEAR}-12-31', freq='D')
				df_complete = pd.DataFrame(date_range, columns=['date'])
				missing_dates = df_complete[~df_complete['date'].isin(observed['date'])]
				total_length = len(df_complete)
				gap_length = len(missing_dates)
				gap_percentage = gap_length / total_length

				if observed.empty:
						message = f'empty obs data for station {file} returning "0" for station\n'
						log_errors(self.model_log_path, message)
						logging.info(message)

						return 0

				elif gap_percentage > 0.1:
						message = f'number of observation data for station {file} has %{gap_percentage} gaps, over %25 gap\treturning "0"\n'
						logging.info(message)
						log_errors(self.model_log_path, message)

						return 0

				else:
						logging.info(f'number of observation data for station {file} has %{gap_percentage} gaps, under %25 gap')

						if gap_length != 0 and gap_percentage < 0.1:
								message = f'{file} has %{gap_percentage} gaps, time series imputation will fill the gaps\n'
								logging.info(message)
								log_errors(self.model_log_path, message)
								observed = filling_observations(df_complete, observed)

						min_observed_year = observed['date'].dt.year.min()
						max_observed_year = observed['date'].dt.year.max()

						simulated = channel_sd_day_data[
								(channel_sd_day_data.yr >= int(min_observed_year))
								& (channel_sd_day_data.yr <= int( max_observed_year + 1))
								& (channel_sd_day_data.gis_id == int(file.split('.')[0].split('_')[0]))
						].reset_index()
						if len(simulated.flo_out.values) == len(observed.streamflow.values):
								simulated.loc[:, 'flo_out'] = self.cms_to_cfs * simulated.flo_out.values
								return creating_plots(
										simulated,
										observed,
										file,
										SCENARIO,
										fig_files_paths,
										self.BASE_PATH,
										self.LEVEL,
										self.VPUID,
										self.NAME,
										self.MODEL_NAME,
										self.stage,
								)
						else:
								message = f'simulations failed {self.NAME}\tsim length: {len(simulated)}\tobs length: {len(observed)}\treturning {self.no_value}\n'
								log_errors(self.model_log_path, message)
								logging.info(message)
								return self.no_value

if __name__ == "__main__":
		BASE_PATH = '/data/MyDataBase/'
		NAME = "40500010102"
		LEVEL = 'huc12'
		VPUID = '0405'
		MODEL_NAME = 'SWAT_gwflow_MODEL'
		START_YEAR = 1997
		END_YEAR = 2015
		nyskip = 3
		no_value = 1e6
		stage = 'verification'

		scenario_TxtInOut = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Scenario_verification_stage_0/')
		streamflow_data_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/streamflow_data/")
		fig_files_paths = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}')
		SCENARIO = 'verification_stage_0'
		evaluator = SwatModelEvaluator(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage)
		objective_function_values = evaluator.model_evaluation(scenario_TxtInOut, streamflow_data_path, fig_files_paths, SCENARIO)