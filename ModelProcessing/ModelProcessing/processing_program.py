import os
import datetime
from ModelProcessing.evaluation import simulate_and_evaluate_swat_model_wrapper
from functools import partial
from ModelProcessing.read_solutions import fetch_new_ranges
import shutil
from skopt.space import Real
from ModelProcessing.utils import delete_previous_runs, delete_previous_figures
from ModelProcessing.utils import read_swat_input_data, write_swat_input_data, read_control_file
from ModelProcessing.utils import update_time, nyskip_define, update_print_prt_file, update_swat_codes_bsn, activate_ET_print
from ModelProcessing.PSO_calibration import PSOOptimizer, save_final_results
from ModelProcessing.sensitivity import SensitivityAnalysis
import time
import pandas as pd
from ModelProcessing.utils import is_cpu_usage_low
import logging
from multiprocessing import Process
from ModelProcessing.logging_utils import get_logger, log_to_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def wrapper_function_model_evaluation(params, username, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, problem, param_files, operation_types, TxtInOut, SCENARIO): 
    return simulate_and_evaluate_swat_model_wrapper(params, username, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, problem, param_files, operation_types, TxtInOut, SCENARIO)

class ProcessingProgram:
    def __init__(self, config):
        self.BASE_PATH = config['BASE_PATH']
        self.LEVEL = config['LEVEL']
        self.VPUID = config['VPUID']
        self.NAME = config['NAME']
        self.MODEL_NAME = config['MODEL_NAME']
        self.START_YEAR = config['START_YEAR']
        self.END_YEAR = config['END_YEAR']
        self.sensitivity_flag = config['sensitivity_flag']
        self.calibration_flag = config['calibration_flag']
        self.verification_flag = config['verification_flag']
        self.nyskip = config['nyskip']
        self.sen_total_evaluations = config['sen_total_evaluations']
        self.sen_pool_size = config['sen_pool_size']
        self.num_levels = config['num_levels']
        self.cal_pool_size = config['cal_pool_size']
        self.max_cal_iterations = config['max_cal_iterations']
        self.termination_tolerance = config['termination_tolerance']
        self.epsilon = config['epsilon']
        self.Ver_START_YEAR = config['Ver_START_YEAR']
        self.Ver_END_YEAR = config['Ver_END_YEAR']
        self.Ver_nyskip = config['Ver_nyskip']
        self.range_reduction_flag = config['range_reduction_flag']
        self.pet = config['pet']
        self.cn = config['cn']
        self.no_value = config['no_value']
        self.verification_samples = config['verification_samples']
        self.username = config['username']  
        # Create a logger specific to this model/station
        self.logger = get_logger(f"{self.MODEL_NAME}.{self.NAME}")
        self.logger.info(f"Initialized processing for {self.MODEL_NAME} - {self.NAME}")

    def global_path_define(self):
        self.original_cal_file         = f'/data/SWATGenXApp/codes/bin/cal_parms_{self.MODEL_NAME}.cal'
        self.general_log_path          = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/log.txt')
        self.model_base                = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/')  
        self.TxtInOut                  = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios/Default/TxtInOut/')
        self.scenarios_path            = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios')
        self.directory_path_si         = os.path.join(self.model_base, f'morris_Si_{self.MODEL_NAME}.csv')
        self.initial_points_path       = os.path.join(self.model_base, f'initial_points_{self.MODEL_NAME}.csv')
        self.initial_values_path       = os.path.join(self.model_base, f'initial_values_{self.MODEL_NAME}.csv')
        self.best_simulation_filename  = os.path.join(self.model_base, f'best_solution_{self.MODEL_NAME}.txt')
        self.lake_path                 = os.path.join(self.model_base, "SWAT_MODEL/Watershed/Shapes/SWAT_plus_lakes.shp")
        self.monthly_cal_figures_path  = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_monthly")
        self.daily_cal_figures_path    = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_daily")
        self.calibration_figures_path  = os.path.join(self.model_base, f"calibration_figures_{self.MODEL_NAME}")
        self.monthly_sen_figures_path  = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_monthly")
        self.daily_sen_figures_path    = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_daily")
        self.cal_file_path             = os.path.join(self.model_base, f'cal_parms_{self.MODEL_NAME}.cal')
        self.local_best_solutions_path = os.path.join(self.model_base, f'local_best_solution_{self.MODEL_NAME}.txt')
        self.model_log_path            = os.path.join(self.model_base, 'log.txt')
        self.gis_folder                = os.path.join(self.model_base, f'{self.MODEL_NAME}/gwflow_gis')
        self.ver_perf_path             = os.path.join(self.model_base, f'recharg_output_{self.MODEL_NAME}/verification_performance_{self.MODEL_NAME}.txt')
        
        self.logger.debug(f"Path definitions complete for {self.MODEL_NAME} - {self.NAME}")

    def PSO_optimization(self, wrapped_model_evaluation):
        self.logger.info(f"Starting PSO optimization for {self.MODEL_NAME} - {self.NAME}")

        opt = PSOOptimizer(
            self.problem, self.username, self.BASE_PATH, self.LEVEL, self.VPUID, self.NAME, self.MODEL_NAME,
            self.model_log_path, self.general_log_path, wrapped_model_evaluation,
            max_it = self.max_cal_iterations, n_particles = self.cal_pool_size, cal_parms = self.cal_parms, 
            best_simulation_filename = self.best_simulation_filename, termination_tolerance = self.termination_tolerance,
            epsilon = self.epsilon, 
            
            
            C1F=0.5, C1I=1, C2I=0.5, C2F=1, Vmax = 0.1, InertiaMin=0.4, InertiaMax=1
        )
        opt.tell()
        best_position, best_score = opt.ask()

        save_final_results(self.best_score, best_position, self.cal_parms, self.best_simulation_filename, self.model_log_path)
        self.logger.info(f"PSO optimization completed for {self.MODEL_NAME} - {self.NAME}")

        return 

    def copy_original_cal_file(self):
        self.logger.debug(f"Copying original calibration file to {self.cal_file_path}")
        if os.path.exists(self.cal_file_path):
            os.remove(self.cal_file_path)
        shutil.copy2(self.original_cal_file, self.model_base)
        self.logger.debug("Calibration file copied successfully")
                
    def clean_up(self):
        self.logger.info("Cleaning up previous runs and figures")
        
        # Delete scenarios and figures
        if self.verification_flag:
            delete_previous_runs(self.scenarios_path)
            recharge_path = os.path.dirname(self.ver_perf_path)    
            if os.path.exists(recharge_path):
                shutil.rmtree(recharge_path)
                self.logger.debug(f"Removed recharge path: {recharge_path}")

        if self.calibration_flag:
            delete_previous_runs(self.scenarios_path)
            delete_previous_figures(self.monthly_cal_figures_path)
            delete_previous_figures(self.daily_cal_figures_path)
            delete_previous_figures(self.calibration_figures_path)
            self.logger.debug("Cleaned calibration figures and runs")

        if self.sensitivity_flag:
            delete_previous_runs(self.scenarios_path)
            delete_previous_figures(self.monthly_sen_figures_path)
            delete_previous_figures(self.daily_sen_figures_path)
            sensitivity_files = ['initial_points', 'morris_Si', 'initial_values']
            for file_name in sensitivity_files:
                file_path = os.path.join(self.model_base, f'{file_name}_{self.MODEL_NAME}.csv')
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Removed file: {file_path}")
                except FileNotFoundError:
                    self.logger.debug(f'File not found: {file_path}')
                except Exception as e:
                    self.logger.error(f'Error removing {file_name}: {e}')
                    
    def remove_lake_parameters(self):
        if not os.path.exists(os.path.join(self.lake_path)):
            message = 'res parameters will be removed'
            self.logger.info(message)
            log_to_file(message, self.model_log_path)
            
            self.cal_parms = self.cal_parms[~self.cal_parms.file_name.isin(['hydrology.res'])].reset_index(drop=True)
            write_swat_input_data(self.model_base, self.cal_parms, f'cal_parms_{self.MODEL_NAME}.cal')
            

    def get_the_best_values(self):
        path = os.path.join(self.BASE_PATH, fr"SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/local_best_solution_{self.MODEL_NAME}.txt")
        
        if not os.path.exists(path):
            self.logger.warning(f'Local best solution for {self.MODEL_NAME}:{self.NAME}:{self.VPUID} does not exist')
            return None
            
        df = pd.read_csv(path, sep = ",")
        df = df.sort_values(by = "best_score", ascending = True)
        
        ## select the best 50 solutions
        df_samples = df.iloc[:150].sample(n=self.verification_samples).reset_index(drop = True)
        df_best = df.iloc[:1].reset_index(drop = True)
        df = pd.concat([df_samples, df_best], axis = 0)
        df.reset_index(drop = True, inplace = True)
        
        # now we need a dictionary to store the parameters and their values
        parameters = df.columns[1:]
        parameters_values = df.iloc[:,1:].values
        parameters_values = [list(x) for x in parameters_values]
        
        solutions = [dict(zip(parameters, x)) for x in parameters_values]
        self.logger.info(f"Retrieved {len(solutions)} best solutions for verification")
        
        return solutions

    def write_verification_performance(self, verification_output):
        try:
            if os.path.exists(self.ver_perf_path):
                with open(self.ver_perf_path, 'a') as file:
                    file.write(verification_output)
                    self.logger.debug(f"Appended verification results to {self.ver_perf_path}")
            else:
                with open(self.ver_perf_path, 'w') as file:
                    file.write(f'NAME,VPUID,MODEL_NAME,START_YEAR,END_YEAR,OBJECTIVE_VALUE\n')
                    file.write(verification_output)
                    self.logger.debug(f"Created verification file {self.ver_perf_path}")
        except Exception as e:
            self.logger.error(f"Error writing verification performance: {e}")
                
    def verification_process_for_name(self, params, SCENARIO, wrapped_model_evaluation):
        self.logger.info(f"Starting verification for scenario {SCENARIO}")
        
        objective_value = 0
        X = list(params.values())
        
        try:
            objective_value = wrapped_model_evaluation(X)
            verification_output = f' {self.NAME},{self.VPUID},{self.MODEL_NAME},{self.Ver_START_YEAR},{self.Ver_END_YEAR},{objective_value}\n'

            self.write_verification_performance(verification_output)
            message = f'Verification {self.MODEL_NAME}:{self.NAME}:{self.VPUID} is completed with objective value {objective_value}'
            self.logger.info(message)
            log_to_file(message, self.model_log_path)
            
            rech_out_folder = os.path.join(self.model_base, f'recharg_output_{self.MODEL_NAME}/{SCENARIO}')
            RESOLUTION = 250
            SOURCE_path = os.path.join(self.model_base, self.MODEL_NAME, f'Scenarios/{SCENARIO}')
            
            try:
                create_recharge_image_for_name(SOURCE_path, self.LEVEL, self.VPUID, self.NAME, RESOLUTION, self.gis_folder, rech_out_folder, self.Ver_START_YEAR, self.Ver_END_YEAR, self.Ver_nyskip)
                recharge_message = f'Verification {self.MODEL_NAME}:{self.NAME}:{self.VPUID} recharge outputs are generated'
                self.logger.info(recharge_message)
                log_to_file(recharge_message, self.model_log_path)
            except Exception as e:
                self.logger.error(f"Error creating recharge image: {e}")
        except Exception as e:
            self.logger.error(f"Error in verification process: {e}")


    def get_cal_parms(self):
        self.logger.info("Loading calibration parameters")
        
        if os.path.exists(self.local_best_solutions_path) and self.range_reduction_flag:
            new_ranges = fetch_new_ranges(self.VPUID, self.NAME, self.LEVEL, self.BASE_PATH, self.MODEL_NAME)
            self.cal_parms = read_swat_input_data(self.model_base, file_name=f'new_cal_parms_{self.MODEL_NAME}.cal')
            self.logger.info("Using reduced parameter ranges from previous solutions")
        else:
            self.cal_parms = read_swat_input_data(self.model_base, file_name=f'cal_parms_{self.MODEL_NAME}.cal')
            self.logger.info("Using original parameter ranges")
                
    def update_model_time_and_printing(self, print_hru=False):
        self.logger.debug("Updating model time settings and print configuration")
        update_time(self.TxtInOut, self.START_YEAR, self.END_YEAR)
        nyskip_define(self.TxtInOut, self.nyskip)
        update_print_prt_file(self.TxtInOut, daily_flow_printing=True, hru_printing=print_hru) 
        activate_ET_print(self.TxtInOut)
                
    def get_space_and_problem(self):
        self.logger.debug("Setting up parameter space and problem definition")
        self.param_files, self.operation_types, self.problem = read_control_file(self.cal_parms)
        self.space = [Real(low, high, name=name) for (low, high), name in zip(self.problem['bounds'], self.problem['names'])]
        self.logger.info(f"Problem defined with {len(self.problem['names'])} parameters")

    def log_errors(self, line_to_append):
        # Using our specialized log_to_file function for model-specific log files
        log_to_file(line_to_append, self.model_log_path)
        log_to_file(line_to_append, self.general_log_path)
        
        # Also log through the structured logger
        self.logger.info(line_to_append)
                
    def process_verification_stage(self):
        self.logger.info("Starting verification stage")
        ## clean up the previous runs
        self.START_YEAR = self.Ver_START_YEAR
        self.END_YEAR = self.Ver_END_YEAR
        print_hru = True
        self.update_model_time_and_printing(print_hru=print_hru)
        list_of_best_solutions = self.get_the_best_values()

        if not list_of_best_solutions:
            self.logger.error("No best solutions found for verification. Aborting verification stage.")
            return

        self.logger.info(f"Verifying {len(list_of_best_solutions)} best solutions")
        
        processes = []
        for i, params in enumerate(list_of_best_solutions):
            SCENARIO = f'verification_stage_{i}'
            self.logger.info(f'Verification iteration {i+1}/{len(list_of_best_solutions)}: {SCENARIO}')
            
            wrapped_model_evaluation = partial(
                wrapper_function_model_evaluation,
                username=self.username,
                nyskip=self.Ver_nyskip, 
                START_YEAR=self.Ver_START_YEAR, 
                END_YEAR=self.Ver_END_YEAR, 
                problem=self.problem, 
                param_files=self.param_files, 
                operation_types=self.operation_types, 
                MODEL_NAME=self.MODEL_NAME, 
                LEVEL=self.LEVEL, 
                VPUID=self.VPUID, 
                NAME=self.NAME,
                TxtInOut=self.TxtInOut, 
                BASE_PATH=self.BASE_PATH, 
                no_value=self.no_value, 
                stage='verification', 
                SCENARIO=SCENARIO
            )
            
            process = Process(target=self.verification_process_for_name, args=(params, SCENARIO, wrapped_model_evaluation))
            process.start()
            processes.append(process)
            
            while not is_cpu_usage_low():
                time.sleep(1)
                
            message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} Verification iteration, number of evaluations {i}'
            if len(processes) >= 50:
                self.log_errors(message)
                self.logger.info("Waiting for batch of verification processes to complete")
                for process in processes:
                    process.join()
                processes = []
                
        # Wait for any remaining processes
        for process in processes:
            process.join()
        
        self.logger.info("Verification stage completed")

    def SWATGenX_SCV(self): 
        self.logger.info(f"Starting SWATGenX_SCV for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}")
        
        try:
            self.global_path_define()
            self.clean_up()
            self.copy_original_cal_file()
            self.get_cal_parms()
            self.remove_lake_parameters()
            self.get_space_and_problem()
            self.update_model_time_and_printing()
            update_swat_codes_bsn(self.TxtInOut, self.pet, self.cn)

            if self.sensitivity_flag:
                self.logger.info("Starting sensitivity analysis")
                self.stage = 'sensitivity'

                wrapped_model_evaluation = partial(
                    wrapper_function_model_evaluation,
                    username=self.username,
                    nyskip=self.nyskip, 
                    START_YEAR=self.START_YEAR, 
                    END_YEAR=self.END_YEAR, 
                    problem=self.problem, 
                    param_files=self.param_files, 
                    operation_types=self.operation_types, 
                    MODEL_NAME=self.MODEL_NAME, 
                    LEVEL=self.LEVEL, 
                    VPUID=self.VPUID, 
                    NAME=self.NAME,
                    TxtInOut=self.TxtInOut, 
                    BASE_PATH=self.BASE_PATH, 
                    no_value=self.no_value, 
                    stage=self.stage, 
                    SCENARIO=None
                )
                
                run_sensitivity_analysis = SensitivityAnalysis(
                    wrapped_model_evaluation, 
                    space=self.space, 
                    problem=self.problem, 
                    LEVEL=self.LEVEL, 
                    VPUID=self.VPUID, 
                    NAME=self.NAME, 
                    MODEL_NAME=self.MODEL_NAME, 
                    sen_total_evaluations=self.sen_total_evaluations, 
                    num_levels=self.num_levels, 
                    n_parallel_jobs=self.sen_pool_size, 
                    general_log_path=self.general_log_path, 
                    model_log_path=self.model_log_path,
                    directory_path_si=self.directory_path_si, 
                    initial_points_path=self.initial_points_path,
                    initial_values_path=self.initial_values_path, 
                    no_value=self.no_value
                )

                run_sensitivity_analysis.sensitivity_analysis_parallel_queue()
                delete_previous_runs(self.scenarios_path)
                self.logger.info("Sensitivity analysis completed")

            if self.calibration_flag:
                self.logger.info("Starting calibration")
                self.stage = 'calibration'
                
                wrapped_model_evaluation = partial(
                    wrapper_function_model_evaluation,
                    username=self.username,
                    nyskip=self.nyskip, 
                    START_YEAR=self.START_YEAR, 
                    END_YEAR=self.END_YEAR, 
                    problem=self.problem, 
                    param_files=self.param_files, 
                    operation_types=self.operation_types, 
                    MODEL_NAME=self.MODEL_NAME, 
                    LEVEL=self.LEVEL, 
                    VPUID=self.VPUID, 
                    NAME=self.NAME,
                    TxtInOut=self.TxtInOut, 
                    BASE_PATH=self.BASE_PATH, 
                    no_value=self.no_value, 
                    stage=self.stage, 
                    SCENARIO=None
                )
                
                self.PSO_optimization(wrapped_model_evaluation)
                delete_previous_runs(self.scenarios_path)
                self.logger.info("Calibration completed")

            if self.verification_flag:
                self.logger.info("Starting verification")
                self.process_verification_stage()
                self.logger.info("Verification completed")
                
            self.logger.info(f"SWATGenX_SCV completed successfully for {self.MODEL_NAME}:{self.NAME}:{self.VPUID}")
            return 'All process done'
            
        except Exception as e:
            self.logger.error(f"Error in SWATGenX_SCV: {e}", exc_info=True)
            return f'Error: {str(e)}'



