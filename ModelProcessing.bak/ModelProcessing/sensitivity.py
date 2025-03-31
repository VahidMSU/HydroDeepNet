from math import ceil
import numpy as np
from SALib.analyze import morris as ma
from SALib.sample import morris as ms
from multiprocessing import Process, Manager, Queue
import time
from ModelProcessing.utils import *
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


class SensitivityAnalysis:
    def __init__(self, wrapped_model_evaluation, space, problem, n_parallel_jobs, sen_total_evaluations, num_levels, LEVEL, NAME, MODEL_NAME, VPUID, no_value, directory_path_si, initial_points_path, initial_values_path, general_log_path, model_log_path):
        self.wrapped_model_evaluation = wrapped_model_evaluation
        self.space = space
        self.problem = problem
        self.n_parallel_jobs = n_parallel_jobs
        self.sen_total_evaluations = sen_total_evaluations
        self.num_levels = num_levels
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.MODEL_NAME = MODEL_NAME
        self.VPUID = VPUID
        self.no_value = no_value
        self.directory_path_si = directory_path_si
        self.initial_points_path = initial_points_path
        self.initial_values_path = initial_values_path
        self.general_log_path = general_log_path
        self.model_log_path = model_log_path

    def model_evaluation_wrapper(self, results_list, point):
        result = self.wrapped_model_evaluation(point)
        results_list.append(result)

    def sensitivity_analysis_parallel_queue(self):
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

        message = f"Morris Sensitivity Analysis: {self.LEVEL}, {self.NAME}, {self.MODEL_NAME}, #Trajectories: {num_trajectories},  #Levels: {self.num_levels}, #Samples: {len(initial_points)}, #chunks {num_chunks}, #Parallel jobs: {self.n_parallel_jobs}\n"
        log_errors(self.model_log_path, message)
        log_errors(self.general_log_path, message)

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
                    message = f'{self.MODEL_NAME}:{self.NAME}:{self.VPUID} Sensitivity iteration, number of evaluations {n_evaluated}'
                    log_errors(self.general_log_path, message)
                    log_errors(self.model_log_path, message)

                # Remove and join finished processes
                for process in active_processes[:]:
                    if not process.is_alive():
                        process.join()
                        active_processes.remove(process)
                        ##remove the related scenario

        except Exception as e:
            log_errors(self.model_log_path, f"Error managing processes: {e}")

        initial_objective_values = np.array(list(results))

        max_val = np.median(initial_objective_values[initial_objective_values < self.no_value])
        initial_objective_values = np.where(initial_objective_values == self.no_value, max_val, initial_objective_values)
        Si = ma.analyze(morris_problem, np.array(initial_points), initial_objective_values, print_to_console=True)

        self.save_sensitivity_analysis(Si, self.directory_path_si)
        self.save_initial_data(initial_points, initial_objective_values, self.initial_points_path, self.initial_values_path)

        return initial_points, initial_objective_values

    def save_sensitivity_analysis(self, results, file_path):
        pd.DataFrame(results).to_csv(file_path)

    def save_initial_data(self, initial_points, initial_objective_values, initial_points_path, initial_values_path):
        np.savetxt(initial_points_path, initial_points, delimiter=',')
        np.savetxt(initial_values_path, initial_objective_values, delimiter=',')

