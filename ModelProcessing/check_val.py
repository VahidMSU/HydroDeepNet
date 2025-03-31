import os 
from ModelProcessing.recharge import recharge_generator_helper
from functools import partial
from multiprocessing import Process
from ModelProcessing.core import process_SCV_SWATGenXModel
from ModelProcessing.utils import is_cpu_usage_low
import time 



#### the code is desinged to performance calibration, validation and recharge generation for each model in the list of NAMES
#### assuming that the models are in the directory /data/MyDataBase/SWATplus_by_VPUID/0000/huc12
#### you can run the code to check if calibrations, validations and recharge generations are done for each model




def get_wrapped_model_evaluator(NAME, MODEL_NAME, VPUID, BASE_PATH, sensitivity_flag, calibration_flag, verification_flag):
	## make a config dictornary
	config = {
		'LEVEL': 'huc12',
		'NAME': NAME,
		'MODEL_NAME': MODEL_NAME,
		'VPUID': VPUID,
		'BASE_PATH': BASE_PATH,
		'sensitivity_flag': sensitivity_flag,
		'calibration_flag': calibration_flag,
		'verification_flag': verification_flag,
		'START_YEAR': 2006,
		'END_YEAR': 2020,
		'nyskip': 3,
		'sen_total_evaluations': 1000,
		'sen_pool_size': 120,
		'num_levels': 10,
		'cal_pool_size': 50,
		'max_cal_iterations': 75,
		'termination_tolerance': 15,
		'epsilon': 0.001,
		'Ver_START_YEAR': 1997,
		'Ver_END_YEAR': 2020,
		'Ver_nyskip': 3,
		'range_reduction_flag': False,
		'pet': 1,
		'cn': 1,
		'no_value': 1e6,
		'verification_samples': 5,
	}
	
	
	return partial(
		process_SCV_SWATGenXModel,
		config,
	)

def check_final_calibration(VPUID, NAME, MODEL_NAME):
    path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/best_solution_{MODEL_NAME}.txt"
    if not os.path.exists(path):
        print(f"{VPUID} {NAME} calibration file does NOT exist")
        return False
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Final best" in line:
                best_solution = float(line.split(": ")[1].split("\n")[0])
                if best_solution >100:
                    print(f"{VPUID} {NAME} calibration is failed with best solution {best_solution}")
                    return False   
                else: 
                    print(f"{VPUID} {NAME} calibration has been done with best solution {best_solution}")
                    return True
    print(f"{VPUID} {NAME} calibration is NOT completed")
    return False

def check_validations(VPUID, NAME, MODEL_NAME):
    flags = []
    for i in range(5):
        path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{MODEL_NAME}/Scenarios/verification_stage_{i}/simulation.out"
        if not os.path.exists(path):
            print(f"{VPUID} {NAME} validation file does NOT exist")
            return False
        with open(path, "r") as f:
            lines = f.readlines()
            flag = False
            for line in lines:
                if "Execution successfully completed" in line:
                    flag = True
                    flags.append(flag)
                    break
        if not flag:
            flags.append(flag)
    if not all(flags):
        print(f"{VPUID} {NAME} validation runs has NOT been completed")
        return False
    else:
        print(f"{VPUID} {NAME} validation runs has been completed")
        return True

def check_recharge_generated(VPUID, NAME, MODEL_NAME):
    path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/recharg_output_SWAT_gwflow_MODEL/verification_stage_0/recharge_1997.tif"
    if not os.path.exists(path):
        print(f"{VPUID} {NAME} recharge file does not exist")
        return False
    else:
        print(f"{VPUID} {NAME} recharge has been generated")
        return True


if __name__ == "__main__":


    """
    
    You can safely run this code to check if the calibration, validation and recharge generation is done for each model

    If the calibration is not done, the code will start the calibration process
    If the calibration is done, the code will start the validation process
    If the validation is done, the code will start the recharge generation process
    
    """

    uncalibrated_models = []
    calibrated_models = []
    unvalidated_models = [] 
    validated_models = []
    BASE_PATH = '/data/MyDataBase/SWATplus_by_VPUID/0000/huc12'
    NAMES = os.listdir(BASE_PATH)
    NAMES.remove("log.txt")
    MODEL_NAME = 'SWAT_gwflow_MODEL'
    VPUID = "0000"  
    processes = []  
    processes = []
    
    for NAME in NAMES:
        print(f"################## {NAME} ##################")   
        perform_calibration = False
        perform_validation = False    
        
        while not is_cpu_usage_low(n_processes=210):
            print("CPU usage is high, waiting for 2 hours")
            time.sleep(120 * 60)

        ### perform calibration if not done
        cal_flag = check_final_calibration("0000", NAME, MODEL_NAME)
        if not cal_flag:
            uncalibrated_models.append(NAME)
            print(f"Calibration is not completed for {NAME}")
            perform_validation = True
            perform_calibration = True
            task = get_wrapped_model_evaluator(NAME, MODEL_NAME, VPUID, "/data/MyDataBase/", False, perform_calibration, perform_validation)
            p = Process(target=task)
            p.start()
            processes.append(p)
            time.sleep(60)
        else:
            calibrated_models.append(NAME)
        
        ### perform validation if calibration is done
        val_flag = check_validations("0000", NAME, MODEL_NAME)
        if cal_flag and not val_flag:
            print(f"Performing validation for {NAME}")
            unvalidated_models.append(NAME)
            perform_validation = True
            perform_calibration = False
            task = get_wrapped_model_evaluator(NAME, MODEL_NAME, VPUID, "/data/MyDataBase/", False, perform_calibration, perform_validation)
            p = Process(target=task)
            p.start()
            processes.append(p)
            time.sleep(5)
        else:
            validated_models.append(NAME)

        ### perform recharge generation if validation is done
        recharge_flag = check_recharge_generated("0000", NAME, MODEL_NAME)
        if cal_flag and val_flag and not recharge_flag:
            print(f"Performing recharge generation for {NAME}")
            recharge_generator_helper(NAME)
            recharge_flag = check_recharge_generated("0000", NAME, MODEL_NAME)
        print(f"Calibration: {cal_flag}, Validation: {val_flag}, Recharge: {recharge_flag}")
        print("\n")
    print(f"No. of calibrated models: {len(calibrated_models)}")
    print(f"No. of uncalibrated models: {len(uncalibrated_models)}")
    print(f"No. of validated models: {len(validated_models)}")
    print(f"No. of unvalidated models: {len(unvalidated_models)}")
        
    for p in processes:
        p.join()
