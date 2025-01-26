import os
import shutil
import subprocess
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from uuid import uuid4 as random_scenario_name_generator
try:
	from ModelProcessing.utils import write_model_parameters
	from ModelProcessing.utils import filling_observations
	from ModelProcessing.SWATGenXLogging import LoggerSetup
	from ModelProcessing.gw_head_comparision import GroundwaterModelAnalysis
	from ModelProcessing.et_comparision import Compare_MODIS_et_SWAT
	from ModelProcessing.yld_compare import evaluate_yield
	from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
except Exception:
	from utils import  write_model_parameters
	from gw_head_comparision import GroundwaterModelAnalysis
	from SWATGenXLogging import LoggerSetup
	from et_comparision import Compare_MODIS_et_SWAT
	from yld_compare import evaluate_yield
	from SWATGenXConfigPars import SWATGenXPaths


def simulate_and_evaluate_swat_model_wrapper(params, config, problem, param_files, operation_types, stage, SCENARIO):

	evaluator = SwatModelEvaluator(config, SCENARIO=SCENARIO, stage=stage)
	
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

		def __init__(self, config, SCENARIO=None, stage=None):
			self.config = config
			self.execution_file = f'{self.config.bin_path}/swatplus'
			self.key = str(random_scenario_name_generator())
			self.SCENARIO = SCENARIO if SCENARIO is not None else f"Scenario_{self.key}"
			self.scenario_TxtInOut = f'{self.config.model_base}/{self.config.MODEL_NAME}/Scenarios/{self.SCENARIO}/'

			self.config.basin_yield_path = f"{self.config.BASE_PATH}/{self.config.VPUID}/huc12/{self.config.NAME}/{self.config.MODEL_NAME}/Scenarios/{self.SCENARIO}/basin_crop_yld_yr.txt"
			self.stage = "random" if stage is None else stage
			
			self.logger = LoggerSetup(self.config.model_log_path, rewrite=False, verbose=True)	
			self.logger = self.logger.setup_logger("ModelProcessing")
			

		@staticmethod
		def nse(observed, simulated):
			
			"""Nash-Sutcliffe Efficiency"""
			observed_mean = observed.mean()
			return 1 - sum((observed - simulated) ** 2) / sum(
				(observed - observed_mean) ** 2
			)
		@staticmethod
		def mape(observed, simulated):
			"""Mean Absolute Percentage Error"""
			return sum(abs((observed - simulated) / observed)) / len(observed)
		@staticmethod
		def pbias(observed, simulated):
			"""Percent Bias"""
			return sum((observed - simulated) / observed) / len(observed) * 100
		@staticmethod
		def rmse(observed, simulated):
			"""Root Mean Square Error"""
			return (sum((observed - simulated) ** 2) / len(observed)) ** 0.5
		@staticmethod
		def kge(observed, simulated):
			"""Kling-Gupta Efficiency"""
			observed_mean = observed.mean()
			simulated_mean = simulated.mean()
			return (
				1
				- (
					(np.corrcoef(observed, simulated)[0, 1] - 1) ** 2
					+ (observed.std() / simulated.std() - 1) ** 2
					+ (observed_mean / simulated_mean - 1) ** 2
				)
				** 0.5
			)
		
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
			shutil.copytree(self.config.original_TxtInOut, self.scenario_TxtInOut, dirs_exist_ok=True)
			shutil.copy(self.execution_file, self.scenario_TxtInOut)

		def define_timeout(self):
			self.logger.info(f"scenario_TxtInOut Path: {self.scenario_TxtInOut}")
			# Read the list of HRUs
			list_of_hrus = pd.read_csv(self.config.hru_new_target, skiprows=1)
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
			parameter_path = f"{self.config.model_base}/CentralParameters.txt"
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
			self.logger.info(f'writing model parameters\n')
			self.save_model_parameters(params, problem)
			write_model_parameters(
								param_files, params, problem, operation_types,
								self.scenario_TxtInOut, self.config.original_TxtInOut
								)
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
				message = f'execution terminated within {duration} hours\n'
			except subprocess.TimeoutExpired:
					message = f'wall-time met: {round(timeout_threshold / (60 * 60), 2)} hours\n'

			self.logger.info(message)	
			return self.model_evaluation()
		

		def get_streamflow_nse(self, streamflow_path, channel_sd_day_path):
			gis_id = os.path.basename(streamflow_path).split("_")[0]
			# sourcery skip: extract-method
			simulated_streamflow = []
			dates = []
			with open(channel_sd_day_path, "r") as f:
				content = f.readlines()
				columns = content[1].split()
				flo_out_column_num = columns.index("flo_out")
				gis_id_column_num = columns.index("gis_id")
				mon_column_num = columns.index("mon")
				day_column_num = columns.index("day")
				year_column_num = columns.index("yr")

				for line in content[2:]:
					values = line.split()
					if values[gis_id_column_num] == gis_id:
						simulated_streamflow.append(float(values[flo_out_column_num]))
						dates.append(f"{values[year_column_num]}-{values[mon_column_num]}-{values[day_column_num]}")
			## make the date 
			dates = pd.to_datetime(dates)    
			simulated = pd.DataFrame({"date": dates, "flo_out": simulated_streamflow})
			cms_to_cfs = 35.3147
			simulated['flo_out'] = simulated['flo_out'] * cms_to_cfs
			observed = pd.read_csv(streamflow_path, parse_dates=['date'], dtype={'streamflow': float})

			df = observed.merge(simulated, on='date', how='inner')

			### NSE
			daily_nse = 1 - (df['flo_out'] - df['streamflow']).pow(2).sum() / (df['streamflow'] - df['streamflow'].mean()).pow(2).sum()
			print(f"NSE: {daily_nse}")

			### resample to monthly and calculate nse
			df = df.set_index('date')
			df_monthly = df.resample('ME').mean()
			nse_monthly = 1 - (df_monthly['flo_out'] - df_monthly['streamflow']).pow(2).sum() / (df_monthly['streamflow'] - df_monthly['streamflow'].mean()).pow(2).sum()
			print(f"NSE monthly: {nse_monthly}")



			self.daily_streamflow_scores(df, gis_id)
			self.monthly_streamflow_scores(df_monthly, gis_id)


			
			return daily_nse + nse_monthly


		def model_evaluation(self):
			
			
			channel_sd_day_path = os.path.join( self.scenario_TxtInOut ,"channel_sd_day.txt")
			files = os.listdir(self.config.streamflow_data_path)
			files = [os.path.join(self.config.streamflow_data_path, file) for file in files if file.endswith("_daily.csv")]

			return self._objectives_eval(files, channel_sd_day_path)
			

		def _objectives_eval(self, files, channel_sd_day_data):

			overal_streamflow_performance = 0
			for file in files:
				if os.path.exists(file):
					objective_ = self.get_streamflow_nse(file, channel_sd_day_data)
					if objective_ == self.config.no_value or np.isnan(objective_):
						continue	
					overal_streamflow_performance += objective_


			##############################################################Overal Objective function#################################################
			objective_function_values = -1*(overal_streamflow_performance)
			if np.isnan(objective_function_values):
				objective_function_values = self.config.no_value
			########################################################################################################################################


			self.logger.info(f'objective_function_values: {objective_function_values}')	
			return objective_function_values

						
		def plot_streamflow(self, observed, simulated, title, time_step, name_score):
			"""Plot the observed and simulated streamflow"""
			# Create the appropriate date range based on the time step
			#print("########DEBUG####### plotting monthly streamflow")
			data_range = pd.date_range(start=f'{self.config.START_YEAR + self.config.nyskip}-01-01', end=f'{self.config.END_YEAR}-12-31', freq='D')
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
			self._extracted_from_plot_streamflow_monthly_25(time_step, name_score, title)

		def daily_streamflow_scores(self, df, title):
				
			"""Calculate daily streamflow scores"""
			## 
			simulated = df[['date', 'flo_out']]
			observed = df[['date', 'streamflow']]

			#print(f"length of observed data: {len(observed)}\tlength of simulated data: {len(simulated)}")
			daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(observed.streamflow.values, simulated.flo_out.values)
			self.write_performance_scores(title, "Daily", daily_nse_value, daily_mape_value, pbias_value, rmse_value, kge_value)
			self.plot_streamflow(observed['streamflow'].values, simulated['flo_out'].values, title, "daily", daily_nse_value)
			self.logger.info(f"{self.config.NAME} {title} Daily Streamflow: NSE: {daily_nse_value:.2f}, MAPE: {daily_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")
			return daily_nse_value
		
		def monthly_streamflow_scores(self, df_monthly, title):
			
			"""Calculate monthly streamflow scores with a more robust approach"""
			simulated_monthly = df_monthly[['flo_out', 'yr', 'mon']]
			observed_monthly = df_monthly[['streamflow', 'yr', 'mon']]
			## re
			simulated_monthly = simulated_monthly.flo_out.values
			observed_monthly = observed_monthly.streamflow.values
			self.logger.info(f"length of observed data: {len(observed_monthly)}\tlength of simulated data: {len(simulated_monthly)}")
			assert len(observed_monthly) == len(simulated_monthly), "Length of observed and simulated monthly streamflow data must be equal"
			# Calculate metrics
			monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value = self.calculate_metrics(observed_monthly, simulated_monthly)
			
			# Write performance scores and plot streamflow
			self.write_performance_scores(title, "Monthly", monthly_nse_value, monthly_mape_value, pbias_value, rmse_value, kge_value)
			self.plot_streamflow_monthly(observed_monthly, simulated_monthly, title, "monthly", monthly_nse_value)
			self.logger.info(f"{self.config.NAME} {title} Monthly Streamflow: NSE: {monthly_nse_value:.2f}, MAPE: {monthly_mape_value:.2f}, PBIAS: {pbias_value:.2f}, RMSE: {rmse_value:.2f}, KGE: {kge_value:.2f}")	
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
			os.makedirs(
				f'{self.config.fig_files_paths}/SF/{self.stage}/{time_step}', exist_ok=True
			)
			plt.savefig(
				f'{self.config.fig_files_paths}/SF/{self.stage}/{time_step}/{name_score:.2f}_{title}_{int(time.time())}.png',
				dpi=300,
			)
			plt.close()

		def write_performance_scores (self, title, time_step, NSE, MPE, PBIAS, RMSE, KGE):
		
			current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			performance_scores_path=os.path.join(self.config.BASE_PATH, f'{self.config.VPUID}/{self.config.LEVEL}/{self.config.NAME}/CentralPerformance.txt')
			if not os.path.exists(performance_scores_path):
				with open(performance_scores_path, 'w') as file:
					#file.write(f'model performance scores. First creation: {current_time}\n')
					file.write(f'key\tTime_of_writing\tstation\ttime_step\tstage\tMODEL_NAME\tNSE\tMPE\tPBIAS\tRMSE\tKGE\n')
					file.write(f'{self.key}\t{current_time}\t{title}\t{time_step}\t{self.stage}\t{self.config.MODEL_NAME}\t{NSE:.3f}\t{MPE:.3f}\t{PBIAS:.3f}\t{RMSE:.3f}\t{KGE:.3f}\n')
			else:
				with open(performance_scores_path, 'a') as file:
					file.write(f'{self.key}\t{current_time}\t{title}\t{time_step}\t{self.stage}\t{self.config.MODEL_NAME}\t{NSE:.3f}\t{MPE:.3f}\t{PBIAS:.3f}\t{RMSE:.3f}\t{KGE:.3f}\n')




def evaluate_scenario(config, stage):
	"""
	Function to evaluate a single scenario for a given SWAT model.
	"""

	SCENARIO = "verification_stage_0"
	stage = "verification"
	evaluator = SwatModelEvaluator(config, SCENARIO=SCENARIO, stage=stage)
	#	vpuid, level, name, model_name, start_year, end_year, nyskip, no_value, stage, SCENARIO=scenario
	#)
	return evaluator.model_evaluation()



import itertools
if __name__ == "__main__":
	from ModelConfig import ModelConfig
	VPUID = '0407'
	NAME = '04127997'
	LEVEL = 'huc12'
	MODEL_NAME = 'SWAT_MODEL'
	
	config = ModelConfig(VPUID=VPUID, 
					  	NAME=NAME,
					    LEVEL=LEVEL, 
						MODEL_NAME=MODEL_NAME)

	# Define the parameters


	stage = 'verification'

	evaluate_scenario(config, stage)


	

	
