import numpy as np
from multiprocessing import Process, Manager
import pyDOE
import time
import matplotlib.pyplot as plt
import os
import psutil
from psutil import virtual_memory
import shutil
import numpy as np
import os
import contextlib

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

def is_cpu_usage_low(intervals=2, n_processes=180, swat_model=True, process_name='swatplus.exe'):

    """Check if the current CPU usage is lower than the threshold."""
    print(time.ctime())
    CPU_USAGE_AVERAGE = psutil.cpu_percent(interval = intervals)
    print('CPU usage:',CPU_USAGE_AVERAGE)
    used_memory = ram_usage()
    print('RAM usage:',round(used_memory,2))
    if swat_model:
        swatplus_process_count = count_processes(process_name = 'swatplus.exe') +  count_processes(process_name = 'swatplus_sno.exe')
        print('number of swatplus processors:', swatplus_process_count)

        return CPU_USAGE_AVERAGE < 80 and used_memory<400 and swatplus_process_count<n_processes
    else:
        process_count = count_processes(process_name = process_name)

        return CPU_USAGE_AVERAGE < 80 and used_memory<350 and process_count<n_processes
    ################################# statistical functions ########################################################################################
def ram_usage():
    # Get memory details
    memory = psutil.virtual_memory()
    # Total memory
    total_memory = memory.total / (1024 ** 3)  # Convert bytes to GB
    print(f"Total Memory: {total_memory:.2f} GB")
    # Used memory
    used_memory = memory.used / (1024 ** 3)  # Convert bytes to GB
    print(f"Used Memory: {used_memory:.2f} GB")
    # Memory usage percentage
    memory_usage_percent = memory.percent
    print(f"Memory Usage: {memory_usage_percent}%")

    return used_memory
from datetime import datetime
def log_errors(filename, line_to_append):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not os.path.exists(filename):
        print(f"Creating log file: {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a') as file:
        file.write(f"{current_time} - {line_to_append}\n")




def delete_previous_runs(directory_path):
    if not os.path.exists(directory_path):
        return
    folders_in_directory = os.listdir(directory_path)
    for folder_name in folders_in_directory:
        try:
            if folder_name not in ['Default']:
                folder_path = os.path.join(directory_path, folder_name)
                if os.path.isdir(folder_path):
                    try:
                        shutil.rmtree(folder_path)
                    except Exception as e:
                        print(f'Not deleted: {folder_path}, due to error: {e}')

        except Exception as e:
            print(f'Not deleted: {folder_path}, due to error: {e}')



def run_initial_evaluation(n_initial, X_initial, GlobalBestScore_collection,BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, model_log_path, general_log_path, wrapped_model_evaluation, cal_parms, best_simulation_filename, n_evaluated=0):

    with Manager() as manager:

        next_points = X_initial
        processes = []
        active_processes = []
        message = f"{MODEL_NAME}:{NAME}:{VPUID} number of initial points: {n_initial}"
        log_errors(general_log_path, message)
        log_errors(model_log_path, message)

        results_list = manager.list([None] * n_initial)  # Shared list for scores

        for index, next_point in enumerate(next_points):
            save_new_point_to_file(next_point, BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME)
            #save_new_point_to_file(next_point,cal_parms, BASE_PATH, LEVEL, NAME, MODEL_NAME)   # this line is for debug
            process = Process(target=model_evaluation_wrapper, args=(index, results_list, next_point, wrapped_model_evaluation))
            process.start()
            processes.append(process)
            active_processes.append(process)
            n_evaluated += 1

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # check if the processes are still active
        for process in active_processes:
            if process.is_alive():
                process.terminate()
                process.join()
                print("Terminated a process")
                message = f"{MODEL_NAME}:{NAME}:{VPUID} Terminated a process"
                log_errors(general_log_path, message)
                log_errors(model_log_path, message)


        # Update scores from the processes
        LocalBestScore = np.array(list(results_list))
        # if does not exists the type is w, otherwise it is a

        save_local_best_parameters(BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME, cal_parms, LocalBestScore, next_points, type_write='w')


        # Update Global bests based on the new scores
        print("Local Best Score:", LocalBestScore)

        message = f"{MODEL_NAME}:{NAME}:{VPUID} Local Best Score: {LocalBestScore}"
        log_errors(general_log_path, message)
        log_errors(model_log_path, message)
        # replace None in LocalBestScore with np.inf
        LocalBestScore = np.where(LocalBestScore is None, np.inf, LocalBestScore)
        GlobalBestIndex = np.argmin(LocalBestScore)
        GlobalBest = X_initial[GlobalBestIndex]
        GlobalBestScore = LocalBestScore[GlobalBestIndex]

        GlobalBestScore_collection.append(GlobalBestScore)

        #print("Global Best Score:", GlobalBestScore)
        message = f"{MODEL_NAME}:{NAME}:{VPUID} Global Best Score: {GlobalBestScore} for initial runs"
        log_errors(general_log_path, message)
        log_errors(model_log_path, message)

        # Save the best parameters to the file
        best_params = GlobalBest
        best_objective_value = GlobalBestScore
        save_current_best(best_params, best_objective_value, cal_parms, best_simulation_filename, model_log_path, general_log_path, VPUID, NAME, MODEL_NAME)
        while not is_cpu_usage_low():
            time.sleep(10)
    # Sort the initial points based on the scores
    sorted_indices = np.argsort(LocalBestScore)
    X_initial_sorted = X_initial[sorted_indices]
    return X_initial_sorted, LocalBestScore, GlobalBest, GlobalBestScore, GlobalBestScore_collection

def save_local_best_parameters(BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, cal_parms, LocalBestScore, next_points, type_write='w'):
    # if the file already exists, do not write the header
    with open(os.path.join(BASE_PATH,'SWATplus_by_VPUID',VPUID,LEVEL,NAME,f'local_best_solution_{MODEL_NAME}.txt'), type_write) as f:
        # the header of the scores is BEST_SCORE
        if type_write=='w':
            f.write('best_score,'+','.join(cal_parms.name.values)+'\n')
        for i in range(len(LocalBestScore)):
            f.write(f"{LocalBestScore[i]}, {','.join([str(x) for x in next_points[i]])}\n")


def update_velocity_by_role(i, X, V, LocalBest, GlobalBest, InertiaWeight, C1, C2, role):

    """ updating velocity based on the role defined for the particle"""

    R1 = np.random.uniform(0, 1, len(X[i]))
    R2 = np.random.uniform(0, 1, len(X[i]))
    if role == "mentee":
        # this randomly turns off the social and cognitive components
        Cse, Sme = (1, 0) if np.random.randint(0,2)==1 else (0, 1)

        V[i] = (InertiaWeight * V[i] +
                C1 * R1 * (GlobalBest - X[i])*Cse +
                C2 * R2 * (LocalBest[i] - X[i])*Sme)
    elif role == "mentor":
        # Increase exploitation: More weight to best known positions
        exploitation_factor = 1.5       # You can adjust this factor
        V[i] = (  InertiaWeight * V[i] +
                C1 * R1 * (LocalBest[i] - X[i]) +
                exploitation_factor * C2 * R2 * (GlobalBest - X[i]))
    elif role == "independent":
        # Balanced search
        exploitation_factor = 1.5
        V[i] = (  InertiaWeight * V[i]
                + exploitation_factor *C1* R1 * (LocalBest[i] - X[i])
                + C2 * R2 * (GlobalBest - X[i]))
    return V[i]

def update_particle(i, X, V, LocalBest, GlobalBest, InertiaWeight, C1, C2, Vmax, MinB, MaxB, role, results_list, wrapped_model_evaluation):

    """ updating velocity and position of particles and evaluating the new position

    """

    V[i] = update_velocity_by_role(i, X, V, LocalBest, GlobalBest, InertiaWeight, C1, C2, role)

    # Apply velocity limits
    V[i] = np.clip(V[i], -Vmax, Vmax)
    X[i] += V[i]

    # Enhanced velocity correction
    for j in range(len(X[i])):
        if X[i][j] < MinB[j]:
            X[i][j] = MinB[j]
            ## randomly reverse the velocity or make it zero
            V[i][j] = 0.5 * V[i][j] if np.random.randint(0,2)==1 else 0
        elif X[i][j] > MaxB[j]:
            X[i][j] = MaxB[j]
            V[i][j] = 0.5 * V[i][j] if np.random.randint(0,2)==1 else 0

    # Evaluate the new position
    score = wrapped_model_evaluation(X[i])

    results_list[i] = (X[i], V[i], score)


class PSOOptimizer:

    def __init__(self, problem, BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, model_log_path, general_log_path, wrapped_model_evaluation, max_it, n_particles, cal_parms, best_simulation_filename, termination_tolerance, epsilon, C1F=0.5, C1I=1, C2I=0.5, C2F=1, Vmax=0.1, InertiaMin=0.4, InertiaMax=1):
        self.problem = problem
        self.max_it = max_it
        self.n_particles = n_particles
        self.C1F = C1F
        self.C1I = C1I
        self.C2I = C2I
        self.C2F = C2F
        self.Vmax = Vmax
        self.InertiaMin = InertiaMin
        self.InertiaMax = InertiaMax
        self.BASE_PATH = BASE_PATH
        self.LEVEL = LEVEL
        self.NAME = NAME
        self.VPUID = VPUID
        self.MODEL_NAME = MODEL_NAME
        self.model_log_path = model_log_path
        self.general_log_path = general_log_path
        self.wrapped_model_evaluation = wrapped_model_evaluation
        self.n_particles = n_particles
        self.NV = len(problem['bounds'])
        self.MinB = np.array([low for low, high in problem['bounds']])
        self.MaxB = np.array([high for low, high in problem['bounds']])
        self.Vmax = Vmax * (self.MaxB - self.MinB)
        self.cal_parms = cal_parms
        self.best_simulation_filename = best_simulation_filename
        self.X = None
        self.V = None
        self.LocalBest = None
        self.GlobalBest = None
        self.GlobalBestScore = np.inf
        self.GlobalBestScore_collection = []
        self.LocalBestScore_collection = []
        self.termination_tolerance = termination_tolerance
        self.epsilon = epsilon

    def tell(self, initial_values=None, initial_points=None):

        n_initial = len(initial_values) if initial_values else self.n_particles

        X_initial = pyDOE.lhs(self.NV, samples=n_initial, criterion='maximin')
        # transfer X_initial to the actual range of the parameters
        for i in range(self.NV):
            X_initial[:, i] = X_initial[:, i] * (self.MaxB[i] - self.MinB[i]) + self.MinB[i]

        message = f"{self.MODEL_NAME}:{self.NAME} number of initial points: {n_initial}"
        log_errors(self.general_log_path, message)
        log_errors(self.model_log_path, message)

        X_initial_sorted, LocalBestScore, GlobalBest, GlobalBestScore, GlobalBestScore_collection = run_initial_evaluation(n_initial, X_initial, self.GlobalBestScore_collection,self.BASE_PATH,
                                                                                                                        self.LEVEL, self.VPUID, self.NAME, self.MODEL_NAME, self.model_log_path,
                                                                                                                        self.general_log_path, self.wrapped_model_evaluation, self.cal_parms,
                                                                                                                        self.best_simulation_filename
                                                                                                                        )

        self.X = select_best_initial_positions(self.n_particles, n_initial, X_initial_sorted, LocalBestScore)
        self.LocalBest = np.copy(self.X)
        self.LocalBestScore = LocalBestScore

        self.GlobalBest = GlobalBest
        self.GlobalBestScore = GlobalBestScore
        self.GlobalBestScore_collection = GlobalBestScore_collection
        message = f"Initial Best Score: {self.GlobalBestScore}"

        self.cleanup_scenario_directory(message)
    def ask(self):
        self.V = np.zeros((self.n_particles, self.NV))
        with Manager() as manager:
            results_list = manager.list([None] * self.n_particles)  # Shared list to store results

            for It in range(self.max_it):
                InertiaWeight = update_InertiaWeight(self.InertiaMax, self.InertiaMin, It, self.max_it)
                processes = []

                for i in range(self.n_particles):
                    role = define_role(self.LocalBestScore, i)
                    C1, C2 = update_C1C2(self.C1F, self.C1I, self.C2F, self.C2I, It, self.max_it)

                    process = Process(target=update_particle, args=(i, self.X, self.V, self.LocalBest, self.GlobalBest, InertiaWeight, C1, C2, self.Vmax, self.MinB, self.MaxB, role, results_list,self.wrapped_model_evaluation))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

                for i in range(self.n_particles):
                    newX, newV, score = results_list[i]
                    self.X[i] = newX
                    self.V[i] = newV
                    if score < self.LocalBestScore[i]:
                        self.LocalBest[i] = self.X[i]
                        self.LocalBestScore[i] = score
                    if score < self.GlobalBestScore:
                        self.GlobalBest = self.X[i]
                        self.GlobalBestScore = score
                        # save the best parameters to the file
                        save_current_best(self.GlobalBest, self.GlobalBestScore, self.cal_parms, self.best_simulation_filename, self.model_log_path, self.general_log_path, self.VPUID, self.NAME, self.MODEL_NAME)
                        save_local_best_parameters(self.BASE_PATH, self.LEVEL,self.VPUID, self.NAME, self.MODEL_NAME, self.cal_parms, self.LocalBestScore, self.LocalBest,type_write='a')
                    self.LocalBestScore_collection.append([i, It, self.LocalBestScore[i]])

                self.GlobalBestScore_collection.append(self.GlobalBestScore)
                #while not is_cpu_usage_low():
                #    time.sleep(10)

                # early stopping based on the number of iterations
                # our criterion is that if the std of the last 10 iterations is less than 0.01, we stop
                # this is a simple criterion and can be changed

                if It > 10 and np.std(self.GlobalBestScore_collection[-self.termination_tolerance:]) < self.epsilon:
                    message = f"{self.MODEL_NAME}:{self.NAME} Early stopping at iteration {It} with std: {np.std(self.GlobalBestScore_collection[-self.termination_tolerance:])}"
                    log_errors(self.general_log_path, message)
                    log_errors(self.model_log_path, message)

                    break

                print("Iteration", It, "Global Best Score:", self.GlobalBestScore)
                message = f"{self.MODEL_NAME}:{self.NAME} Iteration {It} Global Best Score: {self.GlobalBestScore}"
                self.cleanup_scenario_directory(message)
                plt.figure(figsize=(10, 6))
                plt.plot(self.GlobalBestScore_collection, color='b', marker='o', linestyle='-', linewidth=2, markersize=6)
                plt.xlabel('Iterations')
                plt.ylabel('Objective Value')
                plt.title('Global Best Improvement')
                plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
                # make sure the directory exists
                os.makedirs(fr"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/calibration_figures_{self.MODEL_NAME}", exist_ok=True)
                plt.savefig(fr"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/calibration_figures_{self.MODEL_NAME}/GlobalBestImprovement.png", dpi=300)


            save_final_results(self.GlobalBestScore, self.GlobalBest, self.cal_parms, self.best_simulation_filename, self.model_log_path)

            return self.GlobalBest, self.GlobalBestScore

    def cleanup_scenario_directory(self, message):
        log_errors(self.general_log_path, message)
        log_errors(self.model_log_path, message)
        delete_previous_runs(
            f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios"
        )


def save_new_point_to_file(next_point, BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME):
    # save the next point to the file
    path = os.path.join(BASE_PATH,'SWATplus_by_VPUID', VPUID,LEVEL,NAME,f'initial_point_calibration_{MODEL_NAME}.txt')
    print("**********************************************",   path)
    with open(os.path.join(BASE_PATH,'SWATplus_by_VPUID',VPUID,LEVEL,NAME,f'initial_point_calibration_{MODEL_NAME}.txt'), 'a') as f:
    ### write in one line
        f.write(','.join([str(x) for x in next_point])+'\n')

def save_current_best(best_params, best_objective_value, cal_parms, best_simulation_filename, model_log_path, general_log_path,VPUID, NAME, MODEL_NAME):

    with open(best_simulation_filename, 'w') as f:
        for x, y in zip(cal_parms.name.values, best_params):
            f.write(f"{x}, {y:.2f}\n")
        f.write(f"Best objective value: {best_objective_value}\n")

    message = f"{MODEL_NAME}:{NAME}:{VPUID} New best found: {best_objective_value}"
    log_errors(model_log_path, message)
    log_errors(general_log_path, message)


def save_final_results(best_score,best_position,cal_parms, best_simulation_filename, model_log_path):
    best_params = best_position
    best_objective_value = best_score
    with open(best_simulation_filename, 'w') as f:
        for x, y in zip(cal_parms.name.values, best_params):
            f.write(f"{x}, {y:.2f}\n")
        f.write(f"Final best objective value: {best_objective_value}\n")
    log_errors(model_log_path, f"Final calibration completed with best objective value {best_objective_value}")


def define_role(LocalBestScore,i):
    if LocalBestScore[i] < np.percentile(LocalBestScore, 25):
        role = "mentor"
    return (
        "mentee"
        if LocalBestScore[i] > np.percentile(LocalBestScore, 75)
        else "independent"
    )

def update_InertiaWeight(InertiaMax,InertiaMin,It,max_it):
    return InertiaMax - (It / max_it) * (InertiaMax - InertiaMin)

def update_C1C2(C1F,C1I,C2F,C2I,It,max_it):
    C1=((C1F-C1I)*(It/max_it))+C1I
    C2=((C2F-C2I)*(It/max_it))+C2I
    return C1,C2


def select_best_initial_positions(n_particles, n_initial, X_initial, scores):
    # Assuming X_initial is sorted by scores in ascending order
    return X_initial[:n_particles]

def model_evaluation_wrapper(index, results_list, point, wrapped_model_evaluation):
    result = wrapped_model_evaluation(point)
    results_list[index] = result
