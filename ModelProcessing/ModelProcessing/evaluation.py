import os
import shutil
import subprocess
import logging
import pandas as pd
import time
import glob
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import uuid
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from uuid import uuid4 as random_scenario_name_generator
import sys
from multiprocessing import Pool
import os
try:
	from ModelProcessing.utils import log_errors, write_model_parameters
	from ModelProcessing.utils import filling_observations
	from ModelProcessing.gw_head_comparision import GroundwaterModelAnalysis
	from ModelProcessing.et_comparision import Compare_MODIS_et_SWAT
	from ModelProcessing.yld_compare import evaluate_yield
except Exception:
	from utils import log_errors, filling_observations, write_model_parameters
	from gw_head_comparision import GroundwaterModelAnalysis
	from et_comparision import Compare_MODIS_et_SWAT
	from yld_compare import evaluate_yield

### add time to loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def simulate_and_evaluate_swat_model_wrapper(params, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, problem, param_files, operation_types, TxtInOut, SCENARIO):
		
		evaluator = SwatModelEvaluator(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, TxtInOut= TxtInOut, SCENARIO=SCENARIO)
		
		
		return evaluator.simulate_and_evaluate_swat_model(params, problem, param_files, operation_types)


class SwatModelEvaluator:
		"""This class is used to evaluate the SWAT model
		
		Processes: 
		1. Streamflow evaluation in daily and monthly time steps
		2. Groundwater head evaluation in average annual 
		3. ET evaluation in monthly
		4. Overall evaluation of streamflow, groundwater head, and ET
		5. Writing performance scores 
		"""

		def __init__(self, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, TxtInOut=None, SCENARIO=None):
				
				self.execution_file = '/data/MyDataBase/bin/swatplus'
				self.key = str(random_scenario_name_generator())
				self.SCENARIO = SCENARIO if SCENARIO is not None else "Scenario_" + self.key
				self.BASE_PATH = BASE_PATH
				self.VPUID = VPUID
				self.LEVEL = LEVEL
				self.NAME = NAME
				self.MODEL_NAME = MODEL_NAME
				self.START_YEAR = START_YEAR
				self.END_YEAR = END_YEAR
				self.scenario_TxtInOut = f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/{self.SCENARIO}/'
				self.original_TxtInOut = TxtInOut if TxtInOut is not None else f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/'
				self.hru_new_target = f'{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/hru.con'
				self.model_log_path = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/log.txt")
				self.general_log_path = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/log.txt')
				self.fig_files_paths = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/figures_{MODEL_NAME}')
				self.streamflow_data_path = os.path.join(self.BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{self.NAME}/streamflow_data/")
				self.model_log_path = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{str(NAME)}/log.txt")
				self.basin_yield_path = f"/data/MyDataBase/SWATplus_by_VPUID/{self.VPUID}/huc12/{self.NAME}/{self.MODEL_NAME}/Scenarios/{self.SCENARIO}/basin_crop_yld_yr.txt"
				self.nyskip = nyskip
				self.no_value = no_value
				self.stage = "random" if stage is None else stage
				self.cms_to_cfs = 35.3147

		@staticmethod
		def nse(observed, simulated):
				"""Nash-Sutcliffe Efficiency"""
				observed_mean = observed.mean()
				nse = 1 - sum((observed - simulated) ** 2) / sum((observed - observed_mean) ** 2)
				return nse
		@staticmethod
		def mape(observed, simulated):
				"""Mean Absolute Percentage Error"""
				mape = sum(abs((observed - simulated) / observed)) / len(observed)
				return mape
		@staticmethod
		def pbias(observed, simulated):
				"""Percent Bias"""
				pbias = sum((observed - simulated) / observed) / len(observed) * 100
				return pbias
		@staticmethod
		def rmse(observed, simulated):
				"""Root Mean Square Error"""
				rmse = (sum((observed - simulated) ** 2) / len(observed)) ** 0.5
				return rmse
		@staticmethod
		def kge(observed, simulated):
				"""Kling-Gupta Efficiency"""
				observed_mean = observed.mean()
				simulated_mean = simulated.mean()
				kge = 1 - ((np.corrcoef(observed, simulated)[0, 1] - 1) ** 2 + (observed.std() / simulated.std() - 1) ** 2 + (observed_mean / simulated_mean - 1) ** 2) ** 0.5
				return kge
		
		@staticmethod
		def calculate_metrics(observed, simulated):
				"""Calculate NSE, MAPE, PBIAS, RMSE"""
				nse = SwatModelEvaluator.nse(observed, simulated)
				mape = SwatModelEvaluator.mape(observed, simulated)
				pbias = SwatModelEvaluator.pbias(observed, simulated)
				rmse = SwatModelEvaluator.rmse(observed, simulated)
				kge = SwatModelEvaluator.kge(observed, simulated)

				return nse, mape, pbias, rmse, kge
		
		def prepare_scenario_files(self):
				shutil.copytree(self.original_TxtInOut, self.scenario_TxtInOut, dirs_exist_ok=True)
				shutil.copy(self.execution_file, self.scenario_TxtInOut)

		def define_timeout(self):
				# Read the list of HRUs
				list_of_hrus = pd.read_csv(self.hru_new_target, skiprows=1)
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

		def save_model_parameters(self, params, problem):
			params_dict = dict(zip(problem['names'], params))
			parameter_path = f"{self.BASE_PATH}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/CentralParameters.txt"
			os.makedirs(os.path.dirname(parameter_path), exist_ok=True)
			if not os.path.exists(parameter_path):
				with open(parameter_path, 'w') as file:
					# First line is the header
					file.write(f'key {" ".join(problem["names"])}\n')
					file.write(f'{self.key} {" ".join([str(params_dict[name]) for name in problem["names"]])}\n')
			else:
				# If exists, append the new parameters
				with open(parameter_path, 'a') as file:
					# Key and parameters
					file.write(f'{self.key} {" ".join([str(params_dict[name]) for name in problem["names"]])}\n')

		def simulate_and_evaluate_swat_model(self, params, problem, param_files, operation_types):
				self.prepare_scenario_files()
				logging.info(f'writing model parameters for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}\n')
				self.save_model_parameters(params, problem)
				write_model_parameters(param_files, params, problem, operation_types,
														self.scenario_TxtInOut, self.original_TxtInOut)
				start_model_time = time.time()
				timeout_threshold = self.define_timeout()
				try:
						# Run the model in silent mode
						subprocess.run([self.execution_file],
                     cwd=self.scenario_TxtInOut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     timeout=timeout_threshold)
						end_model_time = time.time()
						duration = end_model_time - start_model_time
						duration = round(duration / (60 * 60), 2)
						message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} execution terminated within {duration} hours\n'
				except subprocess.TimeoutExpired:
						message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} wall-time met: {round(timeout_threshold / (60 * 60), 2)} hours\n'
				log_errors(self.model_log_path, message)
				log_errors(self.general_log_path, message)
				logging.info(message)
				return self.model_evaluation()
		
		def trash_empty_files(self, file):
				if not bool(os.path.exists(file) and os.path.getsize(file) != 0):
						message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} {file} is empty, removing the file\n'
						log_errors(self.model_log_path, message)
						logging.info(message)
						os.remove(file)
						return True
				return False
		
		def model_evaluation(self):
				os.chdir(self.scenario_TxtInOut)
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
						#files = [file for file in os.listdir(self.streamflow_data_path) if file.endswith('.csv') and not self.trash_empty_files(file)]
						files = glob.glob(f'{self.streamflow_data_path}*_daily.csv')
						## end with _filled.csv

						if files:
								files = [file for file in files if not self.trash_empty_files(file)]
						else:
								message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No streamflow data exists, returning {self.no_value}\n'
								log_errors(self.model_log_path, message)
								logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No streamflow data exists, returning {self.no_value}')
								return self.no_value
						if channel_sd_day_data.empty:
								message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} SWAT simulation output is empty, returning {self.no_value}\n'
								log_errors(self.model_log_path, message)
								logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} SWAT simulation output is empty, returning {self.no_value}')
								return self.no_value
						else:

								

								overal_streamflow_performance = 0
								for file in files:
										if os.path.exists(file):
											objective_ = self.cal_streamflow_obj_val(file, channel_sd_day_data)
											if objective_ == self.no_value or np.isnan(objective_):
												continue	
											overal_streamflow_performance += objective_
											
								gwflow_evaluator = GroundwaterModelAnalysis(self.NAME, self.MODEL_NAME, self.VPUID, self.LEVEL, self.scenario_TxtInOut, key=self.key)
								if gwflow_evaluator.check_files():
										gwflow_evaluator.process()
										head_nse_score, head_mape_score, head_pbias_score, head_rmse_score, head_kge_score, swl_nse_score, swl_mape_score, swl_pbias_score, swl_rmse_score, swl_kge_score = gwflow_evaluator.calculate_metrics()
										self.write_performance_scores("Head", "AveAnnual", head_nse_score, head_mape_score, head_pbias_score, head_rmse_score, head_kge_score)
										self.write_performance_scores("SWL", "AveAnnual", swl_nse_score, swl_mape_score, swl_pbias_score, swl_rmse_score, swl_kge_score)
										gwflow_evaluator.compare_heads_swl() 
										#gwflow_evaluator.plot_heads(gwflow_evaluator.observed_head, name="observed_heads")
										#gwflow_evaluator.plot_heads(gwflow_evaluator.simulated_heads, name="simulated_heads")
								else:
										print(f"File not found for {self.NAME}")
								et_nse_score, et_mape_score, et_pbias_score, et_rmse_score, et_kge_score = Compare_MODIS_et_SWAT(self.NAME, self.LEVEL, self.VPUID, self.scenario_TxtInOut, key=self.key, stage=self.stage, MODEL_NAME=self.MODEL_NAME).process_ET()
								self.write_performance_scores("ET", "Monthly", et_nse_score, et_mape_score, et_pbias_score, et_rmse_score, et_kge_score)
								
								### compare yield
								eval = evaluate_yield(self.basin_yield_path)

								yld_nse_value, yld_rmse_value, yld_kge_value, yld_mape_value, yld_pbias_value = eval.process_and_evaluate()
								if yld_nse_value == None:
									## assuming there has been no yield simulation in the watershed
									yld_nse_value, yld_rmse_value, yld_kge_value, yld_mape_value, yld_pbias_value = 0, 0, 0, 0, 0
								else:
									self.write_performance_scores("Yield", "Annual", yld_nse_value, yld_mape_value, yld_pbias_value, yld_rmse_value, yld_kge_value)
									eval.plot_yield_comparison(output_dir= self.fig_files_paths)
								
								##############################################################Overal Objective function#################################################
								
								objective_function_values = -1*(overal_streamflow_performance + 0.1*et_nse_score + head_nse_score + swl_nse_score + 0.1*yld_nse_value)

								if np.isnan(objective_function_values):
									objective_function_values = self.no_value	
								########################################################################################################################################
								
								logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} objective_function_values: {objective_function_values}')
								return objective_function_values
				else:
						message = f'No SWAT simulation file {file_path} exists, returning {self.no_value}\n'
						log_errors(self.model_log_path, message)
						logging.info(f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} No SWAT simulation file {file_path} exists, returning {self.no_value}')
						return self.no_value
						
		def plot_streamflow(self, observed, simulated, title, time_step, name_score):
				"""Plot the observed and simulated streamflow"""
				# Create the appropriate date range based on the time step
				#print("########DEBUG####### plotting monthly streamflow")
				data_range = pd.date_range(start=f'{self.START_YEAR + self.nyskip}-01-01', end=f'{self.END_YEAR}-12-31', freq='D')
				nse_score, mape_score, pbias_score, rmse_score , kge_score = self.calculate_metrics(observed, simulated)
				# Plot the observed and simulated streamflow
				fig, ax = plt.subplots(figsize=(12, 6))
				ax.plot(data_range, observed, label='Observed', color='blue', linewidth=1)
				ax.plot(data_range, simulated, label='Simulated', color='red', linewidth=1)
				ax.set_title(f'{title} {self.stage} Streamflow')
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
				plt.annotate(f'NSE: {nse_score:.2f}\nMAPE: {mape_score:.2f}\nPBIAS: {pbias_score:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
				# Save the plot
				#print(f"titles: {title}_{self.stage}_streamflow.png")    
				os.makedirs(f'{self.fig_files_paths}/SF/{self.stage}/{time_step}', exist_ok=True)
				plt.savefig(f'{self.fig_files_paths}/SF/{self.stage}/{time_step}/{name_score:.2f}_{title}_{int(time.time())}.png', dpi=300)
				plt.close()

		def daily_streamflow_scores(self, observed, simulated, title):
				
				"""Calculate daily streamflow scores"""
				## remove nan in values
				simulated_ = simulated.dropna()
				observed_ = observed.dropna()
				simulated_ = simulated_.merge(observed_[['date']], on='date', how='right')
				observed_ = observed_.merge(simulated_[['date']], on='date', how='right')

				#print(f"length of observed data: {len(observed)}\tlength of simulated data: {len(simulated)}")
				daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(observed_.streamflow.values, simulated_.flo_out.values)
				self.write_performance_scores(title, "Daily", daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value)
				self.plot_streamflow(observed['streamflow'].values, simulated['flo_out'].values, title, "daily", daily_nse_value)

				print(f"{self.NAME} {title} Daily Streamflow: NSE: {daily_nse_value:.2f}, MAPE: {daily_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")
		
				return daily_nse_value
		
		def monthly_streamflow_scores(self, observed_monthly, simulated_monthly, title):
			
			"""Calculate monthly streamflow scores with a more robust approach"""
			simulated_monthly = simulated_monthly.merge(observed_monthly[['yr','mon']], on=['yr', 'mon'], how='right')
			observed_monthly = observed_monthly.merge(simulated_monthly[['yr','mon']], on=['yr', 'mon'], how='right')
			simulated_monthly = simulated_monthly.reset_index(drop=True)
			observed_monthly = observed_monthly.reset_index(drop=True)
			## re
			simulated_monthly = simulated_monthly.flo_out.values
			observed_monthly = observed_monthly.streamflow.values
			
			print(f"length of observed data: {len(observed_monthly)}\tlength of simulated data: {len(simulated_monthly)}")
			assert len(observed_monthly) == len(simulated_monthly), "Length of observed and simulated monthly streamflow data must be equal"
			# Calculate metrics
			monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(observed_monthly, simulated_monthly)
			
			# Write performance scores and plot streamflow
			self.write_performance_scores(title, "Monthly", monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value)
			self.plot_streamflow_monthly(observed_monthly, simulated_monthly, title, "monthly", monthly_nse_value)
			
			print(f"{self.NAME} {title} Monthly Streamflow: NSE: {monthly_nse_value:.2f}, MAPE: {monthly_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")
			
			return monthly_nse_value
		def plot_streamflow_monthly(self, observed_monthly, simulated_monthly, title, time_step, name_score):
				"""Plot the observed and simulated streamflow"""
				
				plt.figure(figsize=(12, 6))
				plt.grid(linestyle='--', linewidth=0.5)
				plt.plot(observed_monthly, label='Observed', color='blue', linewidth=1)
				plt.plot(simulated_monthly, label='Simulated', color='red', linewidth=1)
				plt.title(f'{title} {self.stage} Streamflow')
				plt.xlabel('Date')
				plt.ylabel('Streamflow (cfs)')
				plt.legend()
				plt.annotate(f'NSE: {name_score:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
				os.makedirs(f'{self.fig_files_paths}/SF/{self.stage}/{time_step}', exist_ok=True)
				plt.savefig(f'{self.fig_files_paths}/SF/{self.stage}/{time_step}/{name_score:.2f}_{title}_{int(time.time())}.png', dpi=300)
				plt.close()

		def write_performance_scores (self, title, time_step, NSE, MPE, PBIAS, RMSE, KGE):
				## if "_daily" in title:
				if "_daily" in title:
					title = title.replace("_daily", "")
				current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				performance_scores_path=os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/CentralPerformance.txt')
				if not os.path.exists(performance_scores_path):
						with open(performance_scores_path, 'w') as file:
								#file.write(f'model performance scores. First creation: {current_time}\n')
								file.write(f'key\tTime_of_writing\tstation\ttime_step\tstage\tMODEL_NAME\tNSE\tMPE\tPBIAS\tRMSE\tKGE\n')
								file.write(f'{self.key}\t{current_time}\t{title}\t{time_step}\t{self.stage}\t{self.MODEL_NAME}\t{NSE:.3f}\t{MPE:.3f}\t{PBIAS:.3f}\t{RMSE:.3f}\t{KGE:.3f}\n')
				else:
					with open(performance_scores_path, 'a') as file:
						file.write(f'{self.key}\t{current_time}\t{title}\t{time_step}\t{self.stage}\t{self.MODEL_NAME}\t{NSE:.3f}\t{MPE:.3f}\t{PBIAS:.3f}\t{RMSE:.3f}\t{KGE:.3f}\n')

		def read_observed_data(self, file):
				"""Read observed data"""
				
				obs = pd.read_csv(file, parse_dates=['date'])[["date", "streamflow"]]
				assert "date" in obs.columns, "date column must be present in the observed data"
				## replace nan with -1
				obs['streamflow'] = np.where(obs['streamflow'] == -1, np.nan, obs['streamflow'])

				observed_monthly = pd.read_csv(file.replace("_daily.csv", "_monthly.csv"))[['yr', 'mon', 'streamflow']]
				## replace nan with -1
				observed_monthly['streamflow'] = np.where(observed_monthly['streamflow'] == -1, np.nan, observed_monthly['streamflow'])
				file = os.path.basename(file)
				
				observed = obs[(obs.date >= f'{self.START_YEAR + self.nyskip}-01-01') & (obs.date <= f'{self.END_YEAR}-12-31')].reset_index(drop=True)
				observed_monthly = observed_monthly[(observed_monthly.yr >= (self.START_YEAR + self.nyskip)) & (observed_monthly.yr <= self.END_YEAR)].reset_index(drop=True)
				print(f"start date in observed: {observed.date.min()}\tend date in observed: {observed.date.max()}")
				
				observed['streamflow'] = np.where(observed['streamflow'] ==-1, np.nan, observed['streamflow'])

				missing_dates = observed[observed.streamflow.isna() == True]
				total_length = len(observed)
				gap_length = len(missing_dates)
				gap_percentage = gap_length / total_length
				return observed, gap_length, gap_percentage, observed_monthly
		
		def cal_streamflow_obj_val(self, file, channel_sd_day_data):
				observed, gap_length, gap_percentage, observed_monthly = self.read_observed_data(file)
				print(f"length of observed data: {len(observed)}\tlength of observed data: {len(observed)}")
				print(f"gap length: {gap_length}\tgap percentage: {gap_percentage}")
				if observed.empty or "date" not in observed.columns:
						message = f'empty obs data for station {os.path.basename(file)} returning "0" for station\n'
						log_errors(self.model_log_path, message)
						logging.info(message)
						return 0
				elif gap_percentage > 0.1:
						message = f'{os.path.basename(file)} has %{gap_percentage} gaps, over %10 gap\treturning "0"\n'
						logging.info(message)
						log_errors(self.model_log_path, message)
						return 0
				else:
						if gap_length != 0 and gap_percentage < 0.1:
								message = f'{file} has %{gap_percentage} gaps, time series imputation will fill the gaps\n'
								logging.info(message)
								log_errors(self.model_log_path, message)
								df_dates = pd.date_range(start=f'{self.START_YEAR + self.nyskip}-01-01', end=f'{self.END_YEAR}-12-31', freq='D')	
								#observed = filling_observations(df_dates, observed)
								observed['streamflow'] = np.where(observed.streamflow == -1, np.mean(observed.streamflow), observed.streamflow)
								observed_monthly['streamflow'] = np.where(observed_monthly.streamflow == -1, np.mean(observed_monthly.streamflow), observed_monthly.streamflow)
						print(f"file: {file}\tlength of observed data: {len(observed)}\tlength of observed data: {len(observed)}")
						min_observed_year = observed['date'].dt.year.min()
						max_observed_year = observed['date'].dt.year.max()
						simulated = channel_sd_day_data[
								(channel_sd_day_data.yr >= int(min_observed_year))
								& (channel_sd_day_data.yr <= int( max_observed_year + 1))
								& (channel_sd_day_data.gis_id == int(os.path.basename(file).split('.')[0].split('_')[0]))
						].reset_index()
						if len(simulated.flo_out.values) == len(observed.streamflow.values):

								data_range = pd.date_range(start=f'{self.START_YEAR + self.nyskip}-01-01', end=f'{self.END_YEAR}-12-31', freq='D')	
								simulated['date'] = data_range
								simulated = simulated.set_index('date')
								simulated.loc[:, 'flo_out'] = self.cms_to_cfs * simulated.flo_out.values
								simulated_monthly = simulated.resample('ME').sum()		
								simulated_monthly['yr'] = simulated_monthly.index.year	
								simulated_monthly['mon'] = simulated_monthly.index.month	
								daily_streamflow_nse_score = self.daily_streamflow_scores(observed, simulated, os.path.basename(file).split('.')[0])
								monthly_streamflow_nse_score = self.monthly_streamflow_scores(observed_monthly, simulated_monthly, os.path.basename(file).split('.')[0])
								overal_streamflow_nse_score = daily_streamflow_nse_score + monthly_streamflow_nse_score
								
								return overal_streamflow_nse_score
						else:
								message = f'simulations failed {self.NAME}\tsim length: {len(simulated)}\tobs length: {len(observed)}\treturning {self.no_value}\n'
								print(f"gap length: {gap_length}\tgap percentage: {gap_percentage}")
								log_errors(self.model_log_path, message)
								logging.info(message)
								return self.no_value


def evaluate_scenario(base_path, vpuid, level, name, model_name, start_year, end_year, nyskip, no_value, stage, i):
    """
    Function to evaluate a single scenario for a given SWAT model.
    """
    try:
        scenario = f"verification_stage_{i}"
        evaluator = SwatModelEvaluator(
            base_path, vpuid, level, name, model_name, start_year, end_year, nyskip, no_value, stage, SCENARIO=scenario
        )
        return evaluator.model_evaluation()
    except Exception as e:
        return f"Error processing {name}, stage {i}: {e}"


if __name__ == "__main__":

	# Define the parameters for the evaluation
    NAMES = os.listdir('/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/')
    NAMES.remove("log.txt")
    BASE_PATH = '/data/MyDataBase/'
    LEVEL = 'huc12'
    VPUID = '0000'
    MODEL_NAME = 'SWAT_gwflow_MODEL'
    START_YEAR = 1997
    END_YEAR = 2020
    nyskip = 3
    no_value = 1e6
    stage = 'verification'
    num_stages = 5  # Number of verification stages

    # Prepare inputs for parallel processing
    tasks = [
        (BASE_PATH, VPUID, LEVEL, name, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, i)
        for name in NAMES
        for i in range(num_stages)
    ]

    # Use multiprocessing to parallelize the evaluations
    with Pool(processes=10) as pool:
        results = pool.starmap(evaluate_scenario, tasks)

    # Handle or log results as needed
    for idx, result in enumerate(results):
        name, stage_id = tasks[idx][3], tasks[idx][10]
        print(f"Result for {name}, stage {stage_id}: {result}")
