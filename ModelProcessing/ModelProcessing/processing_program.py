import os
import datetime
from ModelProcessing.evaluation import simulate_and_evaluate_swat_model_wrapper
from functools import partial
from ModelProcessing.read_solutions import fetch_new_ranges
import shutil
from skopt.space import Real
from ModelProcessing.utils import delete_previous_runs, delete_previous_figures
from ModelProcessing.utils import read_swat_input_data, write_swat_input_data, read_control_file
from ModelProcessing.utils import update_time, nyskip_define, update_print_prt_file, update_swat_codes_bsn, activate_ET_print, update_gwflow_head_output_times
from ModelProcessing.PSO_calibration import PSOOptimizer, save_final_results
from ModelProcessing.sensitivity import SensitivityAnalysis
import time
import pandas as pd
from ModelProcessing.utils import is_cpu_usage_low
from ModelProcessing.recharge import create_recharge_image_for_name
from ModelProcessing.SWATGenXLogging import LoggerSetup
#from ModelProcessing.get_best_performance import Performance_evaluator
from multiprocessing import Process
from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths


class ProcessingProgram:
	def __init__(self, config):
		self.config = config
		self.clean_up()
		self.copy_original_cal_file()
		self.get_cal_parms()
		self.remove_lake_parameters()
		self.get_space_and_problem()
		self.update_model_time_and_printing()
		update_swat_codes_bsn(self.config.TxtInOut, self.config.pet, self.config.cn)
		self.logger = LoggerSetup(self.config.model_base, rewrite=False, verbose=True)
		self.logger = self.logger.setup_logger("ModelProcessing")


	def PSO_optimization(self, wrapped_model_evaluation):
		#BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, model_log_path, 
		opt = PSOOptimizer(	
			self.problem, 
			self.config, 
			wrapped_model_evaluation,
			self.cal_parms, 
			)
		
		opt.tell()
		
		best_position, best_score = opt.ask()

		save_final_results(self.best_score,best_position, self.cal_parms, self.config.best_simulation_filename, self.config.model_log_path)

		return 

	def copy_original_cal_file(self):
		if os.path.exists(self.config.cal_file_path):
			os.remove(self.config.cal_file_path)
		shutil.copy2(self.config.original_cal_file, self.config.model_base)
					
	def clean_up(self):
			# Delete scenarios and figures
		if self.config.verification_flag:
			delete_previous_runs(self.config.scenarios_path)
			recharge_path = os.path.dirname(self.config.ver_perf_path)	
			if os.path.exists(recharge_path):
				shutil.rmtree(recharge_path)

		if self.config.calibration_flag:
			delete_previous_runs(self.config.scenarios_path)
			delete_previous_figures(self.config.monthly_cal_figures_path)
			delete_previous_figures(self.config.daily_cal_figures_path)
			delete_previous_figures(self.config.calibration_figures_path)

		if self.config.sensitivity_flag:
			delete_previous_runs(self.config.scenarios_path)
			delete_previous_figures(self.config.monthly_sen_figures_path)
			delete_previous_figures(self.config.daily_sen_figures_path)
			sensitivity_files = ['initial_points', 'morris_Si', 'initial_values']
			for file_name in sensitivity_files:
				file_path = os.path.join(self.model_base, f'{file_name}_{self.config.MODEL_NAME}.csv')
				try:
					os.remove(file_path)
				except FileNotFoundError:
					self.logger.info(f'File not found: {file_path}')
				except Exception as e:
					self.logger.info(f'Error removing {file_name}: {e}')
									
	def remove_lake_parameters(self):
		if not os.path.exists(os.path.join(self.config.lake_path)):
			self.logger.info('res parameters will be removed\n')
			self.cal_parms = self.cal_parms[~self.cal_parms.file_name.isin(['hydrology.res'])].reset_index(drop=True)
			write_swat_input_data(self.model_base, self.cal_parms, f'cal_parms_{self.config.MODEL_NAME}.cal')
			

	def get_the_best_values(self):
			
		path = os.path.join(SWATGenXPaths.swatgenx_outlet_path, f"{self.config.VPUID}/{self.config.LEVEL}/{self.config.NAME}/local_best_solution_{self.config.MODEL_NAME}.txt")
		if not os.path.exists(path):
			self.logger.info(f'local best solution for {self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} does not exist')
			return None
		
		df = pd.read_csv(path, sep = ",")
		df = df.sort_values(by = "best_score", ascending = True)
		## select the best 50 solutions
		df_samples = df.iloc[:150].sample(n=self.config.verification_samples).reset_index(drop = True)
		df_best = df.iloc[:1].reset_index(drop = True)
		df = pd.concat([df_samples, df_best], axis = 0)
		df.reset_index(drop = True, inplace = True)
		# now we need a dictionary to store the parameters and their values
		parameters = df.columns[1:]
		parameters_values = df.iloc[:,1:].values
		parameters_values = [list(x) for x in parameters_values]

		return [dict(zip(parameters, x)) for x in parameters_values]

	def write_verification_performance(self, objective_value):
		if os.path.exists(self.config.ver_perf_path):
			with open(self.config.ver_perf_path, 'a') as file:
					file.write(f' {self.config.NAME},{self.config.VPUID},{self.config.MODEL_NAME},{self.config.Ver_START_YEAR},{self.config.Ver_END_YEAR},{objective_value}\n')
		else:
			with open(self.config.ver_perf_path, 'w') as file:
					file.write(f'NAME,VPUID,MODEL_NAME,START_YEAR,END_YEAR,OBJECTIVE_VALUE\n')
					file.write(f' {self.config.NAME},{self.config.VPUID},{self.config.MODEL_NAME},{self.config.Ver_START_YEAR},{self.config.Ver_END_YEAR},{objective_value}\n')
							
	def verification_process_for_name(self, params, SCENARIO, wrapped_model_evaluation):
			
		objective_value = 0
		X = list(params.values())
		objective_value = wrapped_model_evaluation(X)

		self.write_verification_performance(objective_value)
		self.logger.info(f'Verification is completed with objective value {objective_value}\n')
		
		rech_out_folder  = os.path.join(self.config.model_base, f'recharg_output_{self.config.MODEL_NAME}/{SCENARIO}')
		
		create_recharge_image_for_name(self.config, rech_out_folder)

		self.logger.info(f'Verification recharge outputs are generated\n')


	def get_cal_parms(self):
		if os.path.exists(self.config.local_best_solutions_path) and self.config.range_reduction_flag:
			new_ranges = fetch_new_ranges(self.config)
			self.cal_parms = read_swat_input_data(self.config.model_base,  file_name=f'new_cal_parms_{self.config.MODEL_NAME}.cal')
		else:
			self.cal_parms = read_swat_input_data(self.config.model_base,  file_name=f'cal_parms_{self.config.MODEL_NAME}.cal')

	def update_time(self):
			time = read_swat_input_data(self.config.TxtInOut, file_name='time.sim')
			time['yrc_start']=self.config.START_YEAR
			time['yrc_end']=self.config.END_YEAR
			write_swat_input_data(self.config.TxtInOut, time, file_name='time.sim')


	def update_model_time_and_printing(self,print_hru = False):
		self.update_time()
		nyskip_define(self.config.TxtInOut,  self.config.nyskip)
		update_print_prt_file(self.config.TxtInOut, daily_flow_printing=True, hru_printing= print_hru) 
		activate_ET_print(self.config.TxtInOut)
		if self.config.MODEL_NAME == 'SWAT_gwflow_MODEL':
			update_gwflow_head_output_times(self.config.TxtInOut)

		
	def get_space_and_problem(self):
		self.param_files, self.operation_types, self.problem = read_control_file(self.cal_parms)

		self.space = [Real(low, high, name=name) for (low, high), name in zip(self.problem['bounds'], self.problem['names'])]

							
	def process_verification_stage(self):
		## clean up the previous runs
		self.config.START_YEAR = self.config.Ver_START_YEAR
		self.config.END_YEAR = self.config.Ver_END_YEAR
		print_hru = True
		self.update_model_time_and_printing(print_hru = print_hru)
		list_of_best_solutions = self.get_the_best_values()
		self.logger.info(f"Number of best solutions are {len(list_of_best_solutions)}")
		self.logger.info(f"Best solutions are {list_of_best_solutions}")
		
		processes = []
		for i, params in enumerate(list_of_best_solutions):
			SCENARIO = f'verification_stage_{i}'
			self.logger.info(f'start year verification {self.config.Ver_START_YEAR} end year verification {self.config.Ver_END_YEAR} nyskip {self.config.Ver_nyskip}')
			wrapped_model_evaluation = partial (simulate_and_evaluate_swat_model_wrapper, config=self.config, problem = self.problem, param_files = self.param_files, operation_types = self.operation_types, 
																		stage = 'verification', SCENARIO = SCENARIO
																	)
			
			process = Process(target = self.verification_process_for_name, args = (params, SCENARIO, wrapped_model_evaluation))
			process.start()
			processes.append(process)
			while not is_cpu_usage_low():
					time.sleep(1)
			if len(processes) >= 50:
					self.logger.info(f'{self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} Verification iteration, number of evaluations {i}')
					for process in processes:
							process.join()
					processes = []
		for process in processes:
				process.join()            



	def process_sensitivity_stage(self):


		wrapped_model_evaluation = partial (
					simulate_and_evaluate_swat_model_wrapper, 
					problem = self.problem, param_files = self.param_files, operation_types = self.operation_types, 
						stage = 'sensitivity', SCENARIO = None 
											)
		
		run_sensitivity_analysis = SensitivityAnalysis(
								wrapped_model_evaluation, space = self.space, problem = self.problem, 
								n_parallel_jobs = self.config.sen_pool_size, 
								model_log_path = self.config.model_log_path,
														)

		run_sensitivity_analysis.sensitivity_analysis_parallel_queue()

		delete_previous_runs(self.config.scenarios_path)

	def process_calibration_stage(self):

		wrapped_model_evaluation = partial (simulate_and_evaluate_swat_model_wrapper, problem = self.problem, param_files = self.param_files, operation_types = self.operation_types, 
																			stage = 'calibration', SCENARIO = None,
																			config = self.config	
																				)
		self.PSO_optimization(wrapped_model_evaluation)
		delete_previous_runs(self.config.scenarios_path)

