import numpy as np
from multiprocessing import Process, Manager
import pyDOE
import time
from ModelProcessing.utils import is_cpu_usage_low
from ModelProcessing.utils import delete_previous_runs
from ModelProcessing.visualization import plot_global_best_improvement
import os
import logging
from matplotlib import pyplot as plt
from ModelProcessing.config import ModelConfig
from ModelProcessing.logging_utils import get_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Initialize logger for this module
logger = get_logger(__name__)

def run_initial_evaluation(n_initial, X_initial, GlobalBestScore_collection, config, wrapped_model_evaluation, cal_parms, best_simulation_filename, n_evaluated=0):
    """Run initial evaluation of particles and save results
    
    Args:
        n_initial: Number of initial particles
        X_initial: Initial particle positions
        GlobalBestScore_collection: Collection to store global best scores
        config: ModelConfig instance with all configuration parameters
        wrapped_model_evaluation: Function to evaluate the model
        cal_parms: Calibration parameters
        best_simulation_filename: File to save best simulation
        n_evaluated: Number of evaluations already performed
        
    Returns:
        X_initial_sorted, LocalBestScore, GlobalBest, GlobalBestScore, GlobalBestScore_collection
    """
    with Manager() as manager:
        next_points = X_initial
        processes = []
        active_processes = []
        message = f"{config.MODEL_NAME}:{config.NAME}:{config.VPUID} number of initial points: {n_initial}"
        logger.error(message)
        
        # Create shared list for evaluation results
        results_list = manager.list([None] * n_initial)
        
        # Start evaluation processes for each particle
        for index, next_point in enumerate(next_points):
            save_new_point_to_file(next_point, config)
            process = Process(target=model_evaluation_wrapper, 
                             args=(index, results_list, next_point, wrapped_model_evaluation))
            process.start()
            processes.append(process)
            active_processes.append(process)
            n_evaluated += 1
            
        # Wait for all processes to finish
        for process in processes:
            process.join()
            
        # Clean up any hanging processes
        for process in active_processes:
            if process.is_alive():
                process.terminate()
                process.join()
                message = f"{config.MODEL_NAME}:{config.NAME}:{config.VPUID} Terminated a process"
                logger.error(message)
                logging.info(message)
                
        # Get evaluation results
        # Each element in results_list should be a single score value, not a tuple
        LocalBestScore = np.array(list(results_list))
        
        # Log individual scores for debugging
        for i, score in enumerate(LocalBestScore):
            logging.debug(f"Particle {i} initial score: {score}")
            
        # Save the scores and parameter values
        save_local_best_parameters(config, cal_parms, LocalBestScore, next_points, type_write='w')
        
        # Log combined results
        logging.info(f"Local Best Score: {LocalBestScore}")
        message = f"{config.MODEL_NAME}:{config.NAME}:{config.VPUID} Local Best Score: {LocalBestScore}"
        logger.error(message)
        
        # Handle any None or NaN values
        LocalBestScore = np.where(LocalBestScore is None, np.inf, LocalBestScore)
        LocalBestScore = np.where(np.isnan(LocalBestScore), np.inf, LocalBestScore)
        
        # Find global best
        GlobalBestIndex = np.argmin(LocalBestScore)
        GlobalBest = X_initial[GlobalBestIndex]
        GlobalBestScore = LocalBestScore[GlobalBestIndex]
        GlobalBestScore_collection.append(GlobalBestScore)
        
        message = f"{config.MODEL_NAME}:{config.NAME}:{config.VPUID} Global Best Score: {GlobalBestScore} for initial runs"
        logger.error(message)
        logging.info(message)
        
        # Save best parameters
        best_params = GlobalBest
        best_objective_value = GlobalBestScore
        save_current_best(best_params, best_objective_value, cal_parms, best_simulation_filename, config)
        
        # Wait for CPU to cool down
        while not is_cpu_usage_low():
            time.sleep(10)
            
    # Sort particles by score for selection
    sorted_indices = np.argsort(LocalBestScore)
    X_initial_sorted = X_initial[sorted_indices]
    LocalBestScore_sorted = LocalBestScore[sorted_indices]
    
    return X_initial_sorted, LocalBestScore_sorted, GlobalBest, GlobalBestScore, GlobalBestScore_collection

def save_local_best_parameters(config, cal_parms, LocalBestScore, next_points, type_write='w'):
    """Save local best parameters to file
    
    Args:
        config: ModelConfig instance
        cal_parms: Calibration parameters
        LocalBestScore: Scores for each particle
        next_points: Parameter values for each particle
        type_write: Write mode ('w' for write, 'a' for append)
    """
    filepath = os.path.join(config.BASE_PATH, 'SWATplus_by_VPUID', config.VPUID, config.LEVEL, 
                           config.NAME, f'local_best_solution_{config.MODEL_NAME}.txt')
    with open(filepath, type_write) as f:
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
    """Update particle position and velocity, then evaluate the new position"""
    try:
        # Update velocity based on particle's role
        V[i] = update_velocity_by_role(i, X, V, LocalBest, GlobalBest, InertiaWeight, C1, C2, role)
        
        # Apply velocity limits
        V[i] = np.clip(V[i], -Vmax, Vmax)
        
        # Update position
        new_X = X[i] + V[i]
        
        # Handle boundary violations
        for j in range(len(new_X)):
            if new_X[j] < MinB[j]:
                new_X[j] = MinB[j]
                # Randomly reverse velocity or zero it out
                V[i][j] = 0.5 * V[i][j] if np.random.randint(0,2)==1 else 0
            elif new_X[j] > MaxB[j]:
                new_X[j] = MaxB[j]
                V[i][j] = 0.5 * V[i][j] if np.random.randint(0,2)==1 else 0
        
        # Store the updated position
        X[i] = new_X
        
        # Evaluate the new position
        score = wrapped_model_evaluation(X[i])
        
        # Log the result
        logging.debug(f"Particle {i} (role: {role}) evaluated with score: {score}")
        
        # Store result as tuple (position, velocity, score)
        results_list[i] = (X[i], V[i], score)
        
    except Exception as e:
        logging.error(f"Error updating particle {i}: {e}")
        # In case of error, use current position with a high score
        results_list[i] = (X[i], V[i], np.inf)

class PSOOptimizer:
    def __init__(self, problem, wrapped_model_evaluation, config, cal_parms, 
                 C1F=0.5, C1I=1, C2I=0.5, C2F=1, Vmax=0.1, InertiaMin=0.4, InertiaMax=1):
        """Initialize PSO optimizer
        
        Args:
            problem: Problem definition with parameter bounds
            wrapped_model_evaluation: Function to evaluate the model
            config: ModelConfig instance with all configuration parameters
            cal_parms: Calibration parameters dataframe
            C1F, C1I, C2F, C2I: PSO acceleration coefficients
            Vmax: Maximum velocity factor
            InertiaMin, InertiaMax: Inertia weight bounds
        """
        self.problem = problem
        self.config = config
        self.cal_parms = cal_parms
        self.wrapped_model_evaluation = wrapped_model_evaluation
        
        # PSO parameters
        self.C1F = C1F
        self.C1I = C1I
        self.C2I = C2I
        self.C2F = C2F
        self.Vmax = Vmax
        self.InertiaMin = InertiaMin
        self.InertiaMax = InertiaMax
        
        # Problem dimensions
        self.NV = len(problem['bounds'])
        self.MinB = np.array([low for low, high in problem['bounds']])
        self.MaxB = np.array([high for low, high in problem['bounds']])
        self.Vmax = Vmax * (self.MaxB - self.MinB)
        
        # Initialize optimization variables
        self.X = None
        self.V = None
        self.LocalBest = None
        self.GlobalBest = None
        self.GlobalBestScore = np.inf
        self.GlobalBestScore_collection = []
        self.LocalBestScore_collection = []

    def tell(self, initial_values=None, initial_points=None):
        n_initial = len(initial_values) if initial_values else self.config.cal_pool_size
        
        X_initial = pyDOE.lhs(self.NV, samples=n_initial, criterion='maximin')
        # transfer X_initial to the actual range of the parameters
        for i in range(self.NV):
            X_initial[:, i] = X_initial[:, i] * (self.MaxB[i] - self.MinB[i]) + self.MinB[i]
            
        message = f"{self.config.MODEL_NAME}:{self.config.NAME} number of initial points: {n_initial}"
        # Use the properly initialized logger from module level
        logger.info(message)
        X_initial_sorted, LocalBestScore, GlobalBest, GlobalBestScore, GlobalBestScore_collection = \
            run_initial_evaluation(n_initial, X_initial, self.GlobalBestScore_collection, 
                                  self.config, self.wrapped_model_evaluation, self.cal_parms, 
                                  self.config.best_simulation_filename)
        
        self.X = select_best_initial_positions(self.config.cal_pool_size, n_initial, X_initial_sorted, LocalBestScore)
        self.LocalBest = np.copy(self.X)
        self.LocalBestScore = LocalBestScore
        self.GlobalBest = GlobalBest
        self.GlobalBestScore = GlobalBestScore
        self.GlobalBestScore_collection = GlobalBestScore_collection
        message = f"Initial Best Score: {self.GlobalBestScore}"
        self.cleanup_scenario_directory(message)

    def ask(self):
        self.V = np.zeros((self.config.cal_pool_size, self.NV))
        with Manager() as manager:
            results_list = manager.list([None] * self.config.cal_pool_size)  # Shared list to store results

            for It in range(self.config.max_cal_iterations):
                InertiaWeight = update_InertiaWeight(self.InertiaMax, self.InertiaMin, It, self.config.max_cal_iterations)
                processes = []

                for i in range(self.config.cal_pool_size):
                    role = define_role(self.LocalBestScore, i)
                    C1, C2 = update_C1C2(self.C1F, self.C1I, self.C2F, self.C2I, It, self.config.max_cal_iterations)

                    process = Process(target=update_particle, 
                                     args=(i, self.X, self.V, self.LocalBest, self.GlobalBest, 
                                          InertiaWeight, C1, C2, self.Vmax, self.MinB, self.MaxB, 
                                          role, results_list, self.wrapped_model_evaluation))
                    processes.append(process)
                    while not is_cpu_usage_low():
                        logging.info("Waiting for CPU usage to be low")
                        time.sleep(10)
                    process.start()

                for process in processes:
                    process.join()

                for i in range(self.config.cal_pool_size):
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
                        save_current_best(self.GlobalBest, self.GlobalBestScore, self.cal_parms, 
                                         self.config.best_simulation_filename, self.config)
                        save_local_best_parameters(self.config, self.cal_parms, self.LocalBestScore, 
                                                 self.LocalBest, type_write='a')
                    self.LocalBestScore_collection.append([i, It, self.LocalBestScore[i]])

                self.GlobalBestScore_collection.append(self.GlobalBestScore)

                # early stopping based on the number of iterations
                # our criterion is that if the std of the last termination_tolerance iterations is less than epsilon, we stop
                if It > 25 and np.std(self.GlobalBestScore_collection[-self.config.termination_tolerance:]) < self.config.epsilon:
                    message = f"{self.config.MODEL_NAME}:{self.config.NAME} Early stopping at iteration {It} with std: {np.std(self.GlobalBestScore_collection[-self.config.termination_tolerance:])}"
                    logger.error(message)
                    self.cleanup_scenario_directory(message)
                    break

                message = f"{self.config.MODEL_NAME}:{self.config.NAME} Iteration {It} Global Best Score: {self.GlobalBestScore}"
                logging.info(message)
                self.cleanup_scenario_directory(message)
                
                # Use the centralized plotting function
                plot_global_best_improvement(
                    global_best_scores=self.GlobalBestScore_collection, 
                    username=self.config.username, 
                    VPUID=self.config.VPUID, 
                    LEVEL=self.config.LEVEL, 
                    NAME=self.config.NAME, 
                    MODEL_NAME=self.config.MODEL_NAME
                )

            save_final_results(self.GlobalBestScore, self.GlobalBest, self.cal_parms, 
                             self.config.best_simulation_filename, self.config.model_log_path)
                
            self.cleanup_scenario_directory(message)
            return self.GlobalBest, self.GlobalBestScore

    def cleanup_scenario_directory(self, message):
        logger.info(message)
        delete_previous_runs(self.config.scenarios_path)
        
def save_new_point_to_file(next_point, config):
    """Save the next point to the file using the ModelConfig"""
    filepath = os.path.join(config.BASE_PATH, 'SWATplus_by_VPUID', config.VPUID, config.LEVEL, 
                           config.NAME, f'initial_point_calibration_{config.MODEL_NAME}.txt')
    with open(filepath, 'a') as f:
        f.write(','.join([str(x) for x in next_point])+'\n')

def save_current_best(best_params, best_objective_value, cal_parms, best_simulation_filename, config):
    """Save the current best parameters to a file
    
    Args:
        best_params: Best parameter values found
        best_objective_value: Best objective value found
        cal_parms: Calibration parameters dataframe
        best_simulation_filename: File to save best simulation
        config: ModelConfig instance with all configuration parameters
    """
    with open(best_simulation_filename, 'w') as f:
        for x, y in zip(cal_parms.name.values, best_params):
            f.write(f"{x}, {y:.2f}\n")
        f.write(f"Best objective value: {best_objective_value}\n")
    
    message = f"{config.MODEL_NAME}:{config.NAME}:{config.VPUID} New best found: {best_objective_value}"
    logger.error(message)


def save_final_results(best_score, best_position, cal_parms, best_simulation_filename, model_log_path):
    """Save final calibration results to a file"""
    best_params = best_position
    best_objective_value = best_score
    with open(best_simulation_filename, 'w') as f:
        for x, y in zip(cal_parms.name.values, best_params):
            f.write(f"{x}, {y:.2f}\n")
        f.write(f"Final best objective value: {best_objective_value}\n")
    logger.error(f"Final calibration completed with best objective value {best_objective_value}")
        
def define_role(LocalBestScore, i):
    """Define the role of particle based on its performance"""
    if LocalBestScore[i] < np.percentile(LocalBestScore, 25):
        role = "mentor"
    return (
        "mentee"
        if LocalBestScore[i] > np.percentile(LocalBestScore, 75)
        else "independent"
    )

def update_InertiaWeight(InertiaMax, InertiaMin, It, max_it):
    """Update inertia weight based on iteration"""
    return InertiaMax - (It / max_it) * (InertiaMax - InertiaMin)

def update_C1C2(C1F, C1I, C2F, C2I, It, max_it):
    """Update cognitive and social parameters based on iteration"""
    C1 = ((C1F-C1I)*(It/max_it))+C1I
    C2 = ((C2F-C2I)*(It/max_it))+C2I
    return C1, C2


def select_best_initial_positions(n_particles, n_initial, X_initial, scores):
    """Select the best n_particles positions from X_initial based on scores"""
    # Assuming X_initial is sorted by scores in ascending order
    return X_initial[:n_particles]

def model_evaluation_wrapper(index, results_list, point, wrapped_model_evaluation):
    """Wrapper function for model evaluation that properly stores results"""
    try:
        # Evaluate the model with the given point
        result = wrapped_model_evaluation(point)
        # Store just the score (for initial evaluation)
        results_list[index] = result
        logging.debug(f"Evaluated particle {index}, score: {result}")
    except Exception as e:
        logging.error(f"Error evaluating particle {index}: {e}")
        # Use a high value for errors
        results_list[index] = np.inf

