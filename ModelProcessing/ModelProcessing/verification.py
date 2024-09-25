import os
from ModelProcessing.utils import log_errors
from ModelProcessing.utils import *
from ModelProcessing.evaluation import SwatModelEvaluator
from ModelProcessing.recharge import create_recharge_image_for_name
def verification_wrapper(params,problem, param_files, operation_types, SCENARIO, BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, Ver_START_YEAR, Ver_END_YEAR, Ver_nyskip, verification_flag):
    """ this function is a wrapper for the verification process"""
    return verification_process_for_name(params,problem, param_files, operation_types, SCENARIO, BASE_PATH, LEVEL, VPUID, NAME, MODEL_NAME, Ver_START_YEAR, Ver_END_YEAR, Ver_nyskip, verification_flag)

def verification_process_for_name(params,problem, param_files, operation_types,  SCENARIO, BASE_PATH, LEVEL,VPUID, NAME, MODEL_NAME, START_YEAR , END_YEAR, nyskip, verification_flag):
    
        
    no_value = 1e6
    model_base= os.path.join(BASE_PATH, f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/')

    RESOLUTION = 250
    model_log_path   = os.path.join(model_base, "log.txt")  
    
    general_log_path = os.path.join(BASE_PATH,f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/log.txt')
    rech_out_folder  = os.path.join(model_base,f'recharg_output_{MODEL_NAME}/{SCENARIO}')
    gis_folder = os.path.join(model_base,f'{MODEL_NAME}/gwflow_gis')
    verification_performance_path = os.path.join(model_base,f'recharg_output_{MODEL_NAME}/{SCENARIO}/verification_performance_{MODEL_NAME}.txt')
    SOURCE_path = os.path.join(model_base,f'Scenarios/Scenario_{SCENARIO}')

    TxtInOut = os.path.join(model_base,f'{MODEL_NAME}/Scenarios/Default/TxtInOut/')
    
    stage = 'verification'
    evaluator = SwatModelEvaluator(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME, START_YEAR, END_YEAR, nyskip, no_value, stage)
    objective_value = evaluator.simulate_and_evaluate_swat_model(params, problem, param_files, operation_types, TxtInOut, SCENARIO)
    
    verification_output = f' {NAME},{VPUID},{MODEL_NAME},{START_YEAR},{END_YEAR},{objective_value}\n'
    
    if os.path.exists(verification_performance_path):
        with open(verification_performance_path, 'a') as file:
            file.write(verification_output)
    else:
        with open(verification_performance_path, 'w') as file:
            file.write(f'NAME,VPUID,MODEL_NAME,START_YEAR,END_YEAR,OBJECTIVE_VALUE\n')
            file.write(verification_output)
    
    message = f'Verification {MODEL_NAME}:{NAME}:{VPUID} is completed with objective value {objective_value}\n'
    log_errors(model_log_path, message)
    log_errors(general_log_path, message)
    
    try:
        create_recharge_image_for_name(SOURCE_path, LEVEL, VPUID, NAME, RESOLUTION, gis_folder, rech_out_folder, START_YEAR, END_YEAR)
        recharge_message = f'Verification {MODEL_NAME}:{NAME}:{VPUID} recharge outputs are generated\n'

    except Exception as e:    

        recharge_message = f'Verification {MODEL_NAME}:{NAME}:{VPUID} failed to generate recharge outputs due to {e}\n'

    log_errors(model_log_path, recharge_message)
    log_errors(general_log_path, recharge_message)
    
    
    