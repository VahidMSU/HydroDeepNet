import os
from utils import *
import time
from processing_program import SWATGenX_SCV
from multiprocessing import Process


debug = False

print('begin')

BASE_PATH = "/data/MyDataBase/CIWRE-BAE/"

NAMES = os.listdir(os.path.join(BASE_PATH, "SWATplus_by_VPUID/huc12/"))
MODEL_NAME = "SWAT_MODEL"
NAMES.remove('log.txt')
LEVEL = 'huc12'

if __name__ == "__main__":
    for NAME in NAMES:
        being_processed = checking_models_under_processing('swatplus', LEVEL)
        if NAME in being_processed:
            print('Models under calibration', being_processed)
            print(f'{MODEL_NAME}:{NAME}:{VPUID} being processed, passing to a new model')
            continue

        best_solution_path = os.path.join(BASE_PATH, "SWATplus_by_VPUID", LEVEL, NAME, f'best_solution_{MODEL_NAME}.txt')
        if os.path.exists(best_solution_path):
            print(f'{MODEL_NAME}:{NAME}:{VPUID} best solution exists')
            continue
        
        print(f'{MODEL_NAME}:{NAME}:{VPUID} best solution does not exist: {best_solution_path}')

        sen_analysis_path = os.path.join(BASE_PATH, "SWATplus_by_VPUID", LEVEL, NAME, f'morris_Si_{MODEL_NAME}.csv')
        sensitivity_flag = not os.path.exists(sen_analysis_path)

        calibration_flag = True
        cal_pool_size = 100
        sen_pool_size = 180
        START_YEAR = 2002
        END_YEAR = 2007
        nyskip = 2
        sen_total_evaluations = 600
        n_bayesian_simulations = 500

        while not is_cpu_usage_low(intervals=10, n_processes=150):
            being_processed = checking_models_under_processing('swatplus', LEVEL)
            print('Models under calibration', being_processed)
            time.sleep(60)
            print(f'waiting to submit {NAME} model for evaluations\n')

        print(NAME, 'MODEL Processing is initiated')
        
        process = Process(target = SWATGenX_SCV, args = (
                                                                BASE_PATH, NAME, MODEL_NAME,
                                                                LEVEL, START_YEAR,
                                                                END_YEAR, nyskip,
                                                                sensitivity_flag,
                                                                calibration_flag,
                                                                cal_pool_size, sen_pool_size,
                                                                n_bayesian_simulations,
                                                                sen_total_evaluations)
                                                                )

        process.start()
        time.sleep(2*60 * 60)
