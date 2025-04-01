from math import ceil
import numpy as np
from SALib.analyze import morris as ma
from SALib.sample import morris as ms
from multiprocessing import Process, Manager, Queue
import time
from ModelProcessing.utils import *
import pandas as pd
import logging
from ModelProcessing.config import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class SensitivityAnalysis:
    def __init__(self, wrapped_model_evaluation, space, problem, n_parallel_jobs, sen_total_evaluations, num_levels, config):
        """
        Initialize SensitivityAnalysis with ModelConfig
        
        Args:
            wrapped_model_evaluation: Function to evaluate the model
            space: Parameter space definition
            problem: Problem definition with parameter names and bounds
            n_parallel_jobs: Number of parallel jobs to run
            sen_total_evaluations: Total number of evaluations for sensitivity analysis
            num_levels: Number of levels for Morris method
            config: ModelConfig instance with all configuration parameters
        """
        self.wrapped_model_evaluation = wrapped_model_evaluation
        self.space = space
        self.problem = problem
        self.config = config
        self.n_parallel_jobs = n_parallel_jobs
        self.sen_total_evaluations = sen_total_evaluations 
        self.num_levels = num_levels

    def model_evaluation_wrapper(self, results_list, point):
        """Evaluate model at the given point and store result in results_list"""
        result = self.wrapped_model_evaluation(point)
        results_list.append(result)

    def sensitivity_analysis_parallel_queue(self):
        """Run Morris sensitivity analysis with parallel processing"""
        morris_problem = {
            'num_vars': len(self.space),
            'names': self.problem['names'],
            'bounds': self.problem['bounds']
        }
        num_trajectories = self.sen_total_evaluations // (len(self.space) + 1)  # Adjusted num_trajectories

        initial_points = ms.sample(
            problem=morris_problem,
            N=num_trajectories,
            num_levels=self.num_levels,
            local_optimization=True
        )

        num_chunks = ceil(self.sen_total_evaluations / self.n_parallel_jobs)

        message = f"Morris Sensitivity Analysis: {self.config.LEVEL}, {self.config.NAME}, {self.config.MODEL_NAME}, #Trajectories: {num_trajectories},  #Levels: {self.num_levels}, #Samples: {len(initial_points)}, #chunks {num_chunks}, #Parallel jobs: {self.n_parallel_jobs}\n"
        logger.info(message)
        initial_objective_values = []

        manager = Manager()
        results = manager.list()
        points_queue = Queue()

        for point in initial_points:
            points_queue.put(point)

        active_processes = []
        n_evaluated = 0

        try:
            while not points_queue.empty() or active_processes:
                while len(active_processes) < self.n_parallel_jobs and not points_queue.empty():
                    point = points_queue.get_nowait()
                    process = Process(target=self.model_evaluation_wrapper, args=(results, point))
                    process.start()
                    active_processes.append(process)
                    while not is_cpu_usage_low():
                        time.sleep(1)
                    n_evaluated += 1
                    message = f'{self.config.MODEL_NAME}:{self.config.NAME}:{self.config.VPUID} Sensitivity iteration, number of evaluations {n_evaluated}'
                    logger.error(message)

                # Remove and join finished processes
                for process in active_processes[:]:
                    if not process.is_alive():
                        process.join()
                        active_processes.remove(process)

        except Exception as e:
            logger.error(f"Error managing processes: {e}")
        finally:
            initial_objective_values = np.array(list(results))

        # Handle potential NaN or no_value results by replacing with median
        valid_values = initial_objective_values[initial_objective_values < self.config.no_value]
        if len(valid_values) > 0:
            max_val = np.median(valid_values)
            initial_objective_values = np.where(initial_objective_values == self.config.no_value, max_val, initial_objective_values)
        else:
            # If all values are no_value, use a default fallback
            logger.error("Warning: All evaluations returned no_value. Using default values.")
            initial_objective_values = np.ones_like(initial_objective_values)
            
        # Analyze the results
        Si = ma.analyze(morris_problem, np.array(initial_points), initial_objective_values, print_to_console=True)

        self.save_sensitivity_analysis(Si, self.config.directory_path_si)
        self.save_initial_data(initial_points, initial_objective_values, self.config.initial_points_path, self.config.initial_values_path)

        return initial_points, initial_objective_values

    def save_sensitivity_analysis(self, results, file_path):
        """Save sensitivity analysis results to a CSV file"""
        pd.DataFrame(results).to_csv(file_path)
        logger.info(f"Saved sensitivity analysis results to {file_path}")
    def save_initial_data(self, initial_points, initial_objective_values, initial_points_path, initial_values_path):
        """Save initial points and their corresponding objective values"""
        np.savetxt(initial_points_path, initial_points, delimiter=',')
        np.savetxt(initial_values_path, initial_objective_values, delimiter=',')
        logger.info(f"Saved initial points to {initial_points_path}")
