import os
import datetime
from ModelProcessing.evaluation import simulate_and_evaluate_swat_model_wrapper
from functools import partial
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
from ModelProcessing.logging_utils import get_logger
from ModelProcessing.config import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def wrapper_function_model_evaluation(params, config, stage, problem, param_files, operation_types, TxtInOut, SCENARIO): 
    return simulate_and_evaluate_swat_model_wrapper(
        params, 
        config.username, 
        config.BASE_PATH, 
        config.VPUID, 
        config.LEVEL, 
        config.NAME, 
        config.MODEL_NAME, 
        config.START_YEAR if stage != 'verification' else config.Ver_START_YEAR, 
        config.END_YEAR if stage != 'verification' else config.Ver_END_YEAR, 
        config.nyskip if stage != 'verification' else config.Ver_nyskip, 
        config.no_value, 
        stage, 
        problem, 
        param_files, 
        operation_types, 
        TxtInOut, 
        SCENARIO
    )

class ProcessingProgram:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Create a logger specific to this model/station with both central and user-specific log files
        self.logger = get_logger(
            f"{config.MODEL_NAME}.{config.NAME}",
            username=config.username,
            vpuid=config.VPUID,
            level_name=config.LEVEL,
            station_name=config.NAME
        )
        self.logger.info(f"Initialized processing for {config.MODEL_NAME} - {config.NAME}")
        
        # Define original calibration file path (this is a constant not in ModelConfig)
        self.original_cal_file = f'/data/SWATGenXApp/codes/bin/cal_parms_{config.MODEL_NAME}.cal'

    def PSO_optimization(self, wrapped_model_evaluation):
        self.logger.info(f"Starting PSO optimization for {self.config.MODEL_NAME} - {self.config.NAME}")

        opt = PSOOptimizer(
            self.problem, 
            wrapped_model_evaluation,
            self.config,
            self.cal_parms,
            C1F=0.5, C1I=1, C2I=0.5, C2F=1, Vmax=0.1, InertiaMin=0.4, InertiaMax=1
        )
        opt.tell()
        best_position, best_score = opt.ask()

        save_final_results(best_score, best_position, self.cal_parms, self.config.best_simulation_filename, self.config.model_log_path)
        self.logger.info(f"PSO optimization completed for {self.config.MODEL_NAME} - {self.config.NAME}")

        return 

    def copy_original_cal_file(self):
        self.logger.debug(f"Copying original calibration file to {self.config.cal_file_path}")
        
        # Create the directory structure if it doesn't exist
        if not os.path.exists(self.config.model_base):
            self.logger.info(f"Creating directory: {self.config.model_base}")
            os.makedirs(self.config.model_base, exist_ok=True)
            
        if os.path.exists(self.config.cal_file_path):
            os.remove(self.config.cal_file_path)
        shutil.copy2(self.original_cal_file, self.config.model_base)
        self.logger.debug("Calibration file copied successfully")
                
    def clean_up(self):
        """
        Clean up all previous runs, figures, and output files to ensure a clean state
        before starting any analysis (sensitivity, calibration, or verification).
        """
        self.logger.info("Performing comprehensive cleanup of previous runs and outputs")
        
        # Clean scenarios directory - this is common for all operations
        if os.path.exists(self.config.scenarios_path):
            delete_previous_runs(self.config.scenarios_path)
            self.logger.debug(f"Cleaned previous run scenarios at {self.config.scenarios_path}")
        
        # Clean all figure directories
        figure_paths = [
            self.config.monthly_cal_figures_path,
            self.config.daily_cal_figures_path, 
            self.config.calibration_figures_path,
            self.config.monthly_sen_figures_path,
            self.config.daily_sen_figures_path
        ]
        
        for path in figure_paths:
            if os.path.exists(path):
                delete_previous_figures(path)
                self.logger.debug(f"Cleaned figures at {path}")
        
        # Clean all sensitivity analysis files
        sensitivity_files = ['initial_points', 'morris_Si', 'initial_values']
        for file_name in sensitivity_files:
            file_path = os.path.join(self.config.model_base, f'{file_name}_{self.config.MODEL_NAME}.csv')
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Removed file: {file_path}")
            except Exception as e:
                self.logger.error(f'Error removing {file_name}: {e}')
        
        # Clean verification output directory
        recharge_path = os.path.join(self.config.model_base, f'recharg_output_{self.config.MODEL_NAME}')
        if os.path.exists(recharge_path):
            shutil.rmtree(recharge_path)
            self.logger.debug(f"Removed recharge output path: {recharge_path}")
        
        # Clean best solution and local best solution files
        solution_files = [
            self.config.best_simulation_filename,
            self.config.local_best_solutions_path
        ]
        for solution_file in solution_files:
            if os.path.exists(solution_file):
                os.remove(solution_file)
                self.logger.debug(f"Removed solution file: {solution_file}")
        
        # Clean any model log files that might exist
        if os.path.exists(self.config.model_log_path):
            with open(self.config.model_log_path, 'w') as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting new model run\n")
            self.logger.debug(f"Reset model log file: {self.config.model_log_path}")
        
        # Clean any additional temporary files that might be in the model base directory
        temp_files = [
            f'initial_point_calibration_{self.config.MODEL_NAME}.txt',
            f'verification_performance_{self.config.MODEL_NAME}.txt'
        ]
        for temp_file in temp_files:
            temp_path = os.path.join(self.config.model_base, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                self.logger.debug(f"Removed temporary file: {temp_path}")
        
        self.logger.info("Cleanup completed - workspace is ready for a new analysis")
                
    def remove_lake_parameters(self):
        if not os.path.exists(self.config.lake_path):
            message = 'res parameters will be removed'
            self.logger.info(message)
            
            self.cal_parms = self.cal_parms[~self.cal_parms.file_name.isin(['hydrology.res'])].reset_index(drop=True)
            write_swat_input_data(self.config.model_base, self.cal_parms, f'cal_parms_{self.config.MODEL_NAME}.cal')
            

    def get_the_best_values(self):
        path = self.config.local_best_solutions_path
        
        if not os.path.exists(path):
            self.logger.warning(f'Local best solution for {self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} does not exist')
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
        
        solutions = [dict(zip(parameters, x)) for x in parameters_values]
        self.logger.info(f"Retrieved {len(solutions)} best solutions for verification")
        
        return solutions

    def write_verification_performance(self, verification_output):
        try:
            if os.path.exists(self.config.ver_perf_path):
                with open(self.config.ver_perf_path, 'a') as file:
                    file.write(verification_output)
                    self.logger.debug(f"Appended verification results to {self.config.ver_perf_path}")
            else:
                with open(self.config.ver_perf_path, 'w') as file:
                    file.write(f'NAME,VPUID,MODEL_NAME,START_YEAR,END_YEAR,OBJECTIVE_VALUE\n')
                    file.write(verification_output)
                    self.logger.debug(f"Created verification file {self.config.ver_perf_path}")
        except Exception as e:
            self.logger.error(f"Error writing verification performance: {e}")
                
    def verification_process_for_name(self, params, SCENARIO, wrapped_model_evaluation):
        self.logger.info(f"Starting verification for scenario {SCENARIO}")
        
        objective_value = 0
        X = list(params.values())
        
        try:
            objective_value = wrapped_model_evaluation(X)
            verification_output = f' {self.config.NAME},{self.config.VPUID},{self.config.MODEL_NAME},{self.config.Ver_START_YEAR},{self.config.Ver_END_YEAR},{objective_value}\n'

            self.write_verification_performance(verification_output)
            message = f'Verification {self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} is completed with objective value {objective_value}'
            self.logger.info(message)
            
            rech_out_folder = os.path.join(self.config.model_base, f'recharg_output_{self.config.MODEL_NAME}/{SCENARIO}')
            RESOLUTION = 250
            SOURCE_path = os.path.join(self.config.model_base, self.config.MODEL_NAME, f'Scenarios/{SCENARIO}')
            
        except Exception as e:
            self.logger.error(f"Error in verification process: {e}")


    def get_cal_parms(self):
        self.logger.info("Loading calibration parameters")
        self.cal_parms = read_swat_input_data(self.config.model_base, file_name=f'cal_parms_{self.config.MODEL_NAME}.cal')
        self.logger.info("Using original parameter ranges")
            
    def update_model_time_and_printing(self, print_hru=False):
        self.logger.debug("Updating model time settings and print configuration")
        update_time(self.config.TxtInOut, self.config.START_YEAR, self.config.END_YEAR)
        nyskip_define(self.config.TxtInOut, self.config.nyskip)
        update_print_prt_file(self.config.TxtInOut, daily_flow_printing=True, hru_printing=print_hru) 
        activate_ET_print(self.config.TxtInOut)
                
    def get_space_and_problem(self):
        self.logger.debug("Setting up parameter space and problem definition")
        self.param_files, self.operation_types, self.problem = read_control_file(self.cal_parms)
        self.space = [Real(low, high, name=name) for (low, high), name in zip(self.problem['bounds'], self.problem['names'])]
        self.logger.info(f"Problem defined with {len(self.problem['names'])} parameters")
                
    def process_verification_stage(self):
        self.logger.info("Starting verification stage")
        
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
                config=self.config,
                problem=self.problem, 
                param_files=self.param_files, 
                operation_types=self.operation_types, 
                TxtInOut=self.config.TxtInOut,
                stage='verification', 
                SCENARIO=SCENARIO
            )
            
            process = Process(target=self.verification_process_for_name, args=(params, SCENARIO, wrapped_model_evaluation))
            process.start()
            processes.append(process)
            
            while not is_cpu_usage_low():
                time.sleep(1)
                
            message = f'{self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} Verification iteration, number of evaluations {i}'
            if len(processes) >= 50:
                self.logger.info(message)
                self.logger.info("Waiting for batch of verification processes to complete")
                for process in processes:
                    process.join()
                processes = []
                
        # Wait for any remaining processes
        for process in processes:
            process.join()
        
        self.logger.info("Verification stage completed")

    def SWATGenX_SCV(self): 
        self.logger.info(f"Starting SWATGenX_SCV for {self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID}")
        
        try:
            self.copy_original_cal_file()
            self.clean_up()
            self.get_cal_parms()
            self.remove_lake_parameters()
            self.get_space_and_problem()
            self.update_model_time_and_printing()
            update_swat_codes_bsn(self.config.TxtInOut, self.config.pet, self.config.cn)

            if self.config.sensitivity_flag:
                self.logger.info("Starting sensitivity analysis")
                stage = 'sensitivity'

                wrapped_model_evaluation = partial(
                    wrapper_function_model_evaluation,
                    config=self.config,
                    problem=self.problem, 
                    param_files=self.param_files, 
                    operation_types=self.operation_types, 
                    TxtInOut=self.config.TxtInOut,
                    stage=stage, 
                    SCENARIO=None
                )
                
                run_sensitivity_analysis = SensitivityAnalysis(
                    wrapped_model_evaluation=wrapped_model_evaluation, 
                    space=self.space, 
                    problem=self.problem, 
                    n_parallel_jobs=self.config.sen_pool_size,
                    sen_total_evaluations=self.config.sen_total_evaluations, 
                    num_levels=self.config.num_levels, 
                    config=self.config
                )

                run_sensitivity_analysis.sensitivity_analysis_parallel_queue()
                delete_previous_runs(self.config.scenarios_path)
                self.logger.info("Sensitivity analysis completed")

            if self.config.calibration_flag:
                self.logger.info("Starting calibration")
                stage = 'calibration'
                
                wrapped_model_evaluation = partial(
                    wrapper_function_model_evaluation,
                    config=self.config,
                    problem=self.problem, 
                    param_files=self.param_files, 
                    operation_types=self.operation_types, 
                    TxtInOut=self.config.TxtInOut,
                    stage=stage, 
                    SCENARIO=None
                )
                
                self.PSO_optimization(wrapped_model_evaluation)
                delete_previous_runs(self.config.scenarios_path)
                self.logger.info("Calibration completed")

            if self.config.verification_flag:
                self.logger.info("Starting verification")
                self.process_verification_stage()
                self.logger.info("Verification completed")
                
            self.logger.info(f"SWATGenX_SCV completed successfully for {self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID}")
            return 'All process done'
            
        except Exception as e:
            self.logger.error(f"Error in SWATGenX_SCV: {e}", exc_info=True)
            return f'Error: {str(e)}'



