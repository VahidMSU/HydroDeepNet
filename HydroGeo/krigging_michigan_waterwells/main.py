import subprocess
from kriging_subprocess import run_script
from multiprocessing import Pool
parameter = sys.argv[1]

if __name__ == "__main__":
    parameters = ['SWL', 'TRANSMSV_2', 'TRANSMSV_1', 'AQ_THK_1', 'AQ_THK_2', 'V_COND_1', 'V_COND_2', 'H_COND_2', 'H_COND_1']
    with Pool(3) as pool:  # Limiting to 10 workers
        pool.map(run_script, parameters)
