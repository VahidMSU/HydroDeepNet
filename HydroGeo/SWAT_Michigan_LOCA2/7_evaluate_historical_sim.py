
import sys
import os
import shutil

def remove_scores(VPUID, NAME, stage):
    logfile = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{stage}_performance_scores.txt"
    if os.path.exists(logfile):
        ## remove log.txt
        os.remove(logfile)
    historical_figs= f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL_historical_daily"
    if os.path.exists(historical_figs):
        shutil.rmtree(historical_figs)
    historical_figs= f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL_historical_monthly"
    if os.path.exists(historical_figs):
        shutil.rmtree(historical_figs)
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('/data/MyDataBase/SWATGenXAppData/codes/ModelProcessing/')
from ModelProcessing.cc_evaluation import SwatModelEvaluator

def evaluate_model(base_path, vpuid, level, name, model_name, start_year, end_year, nyskip, no_value, stage, cc_model):
    scenario_TxtInOut = os.path.join(base_path, f'SWATplus_by_VPUID/{vpuid}/{level}/{name}/climate_change_models/{cc_model}')
    streamflow_data_path = os.path.join(base_path, f"SWATplus_by_VPUID/{vpuid}/{level}/{name}/streamflow_data/")
    fig_files_paths = os.path.join(base_path, f'SWATplus_by_VPUID/{vpuid}/{level}/{name}')
    evaluator = SwatModelEvaluator(base_path, vpuid, level, name, model_name, start_year, end_year, nyskip, no_value, stage)
    return evaluator.model_evaluation(scenario_TxtInOut, streamflow_data_path, fig_files_paths, cc_model)

if __name__ == "__main__":
    BASE_PATH = 'E:/MyDataBase'
    NAME = "40500010102"
    LEVEL = 'huc12'
    MODEL_NAME = 'SWAT_gwflow_MODEL'
    START_YEAR = 1997
    END_YEAR = 2015
    nyskip = 3
    no_value = 1e6
    stage = 'historical'
    VPUIDs = os.listdir(os.path.join(BASE_PATH, 'SWATplus_by_VPUID'))
    parallel = True

    with ProcessPoolExecutor(max_workers=61) as executor:
        futures = []
        for VPUID in VPUIDs:

            NAMES = os.listdir(os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}'))
            if 'log.txt' in NAMES:
                NAMES.remove('log.txt')
            for NAME in NAMES:
                remove_scores(VPUID, NAME, stage)
                climate_change_path = os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/climate_change_models/')
                cc_models = os.listdir(climate_change_path)
                cc_models = [cc_model for cc_model in cc_models if not cc_model.endswith('jpeg')]
                #for cc_model in cc_models:
                #    if parallel:
                #        futures.append(executor.submit(evaluate_model, BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, cc_model))
                #    else:

                #        evaluate_model(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage, cc_model)

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Completed evaluation: {result}")
            except Exception as e:
                print(f"Error during evaluation: {e}")
