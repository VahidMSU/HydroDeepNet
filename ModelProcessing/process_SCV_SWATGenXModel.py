from ModelProcessing.core import process_SCV_SWATGenXModel
from ModelProcessing.utils import copy_files
import os
import shutil
import time
from functools import partial
from multiprocessing import Process
from ModelProcessing.utils import delete_previous_runs
import logging
from ModelProcessing.utils import is_cpu_usage_low
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='/home/rafieiva/MyDataBase/codes/ModelProcessing/log.txt')
"""
/***************************************************************************
		SWATGenX
														-------------------
				begin                : 2023-05-15
				copyright            : (C) 2024 by Vahid Rafiei
				email                : rafieiva@msu.edu
***************************************************************************/

/***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************/
"""


###################### Note ##############################
## this script is designed to run Michigan models ##
###################### Note ##############################
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
		'START_YEAR': 2007,
		'END_YEAR': 2020,
		'nyskip': 3,
		'sen_total_evaluations': 1000,
		'sen_pool_size': 120,
		'num_levels': 10,
		'cal_pool_size': 50,
		'max_cal_iterations': 75,
		'termination_tolerance': 10,
		'epsilon': 0.0001,
		'Ver_START_YEAR': 1997,
		'Ver_END_YEAR': 2020,
		'Ver_nyskip': 3,
		'range_reduction_flag': False,
		'pet': 1,
		'cn': 1,
		'no_value': 1e6,
		'verification_samples': 3,
	}
	
	
	return partial(
		process_SCV_SWATGenXModel,
		config,
	)



def check_final_calibration(VPUID, NAME, MODEL_NAME):
	path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/best_solution_{MODEL_NAME}.txt"
	if not os.path.exists(path):
		return False
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if "Final best objective value" in line:
				return True

def check_verification_existance(VPUID, NAME, MODEL_NAME):
	path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{MODEL_NAME}/Scenarios/Scenario_verification_stage_0"
	if not os.path.exists(path):
		return False

	files = os.listdir(path)
	return len(files) != 0

def check_last_modification_date(path):
		## if the modification is less than 24 hours ago, return False
		## otherwise return True
		## this is to avoid running the same model multiple times
		if os.path.exists(path):
			modification_time = os.path.getmtime(path)
			current_time = time.time()
			if current_time - modification_time < 86400:
				return False
		return True

def check_sensitivity_existance(VPUID, NAME, MODEL_NAME):
	path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/morris_Si_{MODEL_NAME}.csv"
	return bool(os.path.exists(path))

def check_source_target_existance(VPUID, NAME, MODEL_NAME):
	source_path = f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/"
	target_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/"

	if os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/gwflow.input") and not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/gwflow.input"):
		streamflow_path = os.path.join(source_path, 'streamflow_data')
		if os.path.exists(streamflow_path) and os.path.exists(os.path.join(source_path, MODEL_NAME)):
			os.makedirs(os.path.join(target_path, MODEL_NAME), exist_ok=True)
			logging.info(f"Copying {VPUID} {NAME}.........")
			shutil.copytree(os.path.join(source_path, MODEL_NAME), os.path.join(target_path, MODEL_NAME), dirs_exist_ok=True)
			shutil.copytree(os.path.join(source_path, 'streamflow_data'), os.path.join(target_path, 'streamflow_data'), dirs_exist_ok=True)
		else:
			try:
				shutil.rmtree(os.path.join(target_path, MODEL_NAME), ignore_errors=True)
				shutil.rmtree(os.path.join(target_path, 'streamflow_data'), ignore_errors=True)
			except Exception as e:
				logging.info(e)
				return False


	path = os.path.join(source_path, 'streamflow_data')
	if os.path.exists(path):
		shutil.rmtree(os.path.join(target_path, 'streamflow_data'), ignore_errors=True)
		os.makedirs(os.path.join(target_path, 'streamflow_data'))
		files = os.listdir(path)
		for file in files:
			shutil.copy2(os.path.join(path, file), os.path.join(target_path, 'streamflow_data'))
	else:
		logging.info(f"#####################{VPUID} {NAME} SWAT_input does not have streamflow_data")
		### remove if from target if exists
		if os.path.exists(target_path):
			shutil.rmtree(target_path)
			logging.info(f"#####################{VPUID} {NAME} streamflow_data removed from target")
		return False

	return True


def check_validation_existance(VPUID, NAME, MODEL_NAME):
	path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{MODEL_NAME}/Scenarios/Scenario_validation_stage_0"
	if not os.path.exists(path):
		return False

	files = os.listdir(path)
	return len(files) != 0

if __name__ == "__main__":

		BASE_PATH = '/data/MyDataBase/'

		# Check if the path exists
		if os.path.exists(BASE_PATH):
				logging.info('Path exists')
		else:
				raise ValueError('Path does not exist')

		INPUT_PATH = '/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12'
		OUTPUT_PATH = '/data/MyDataBase/SWATplus_by_VPUID'
		MODEL_NAME = 'SWAT_gwflow_MODEL'
		VPUID = '0000'
		processes = []

		while True:
				NAMES = os.listdir(INPUT_PATH)
				if 'log.txt' in NAMES:
						NAMES.remove('log.txt')
				for NAME in NAMES:
						if len(NAME) > 10:
								continue
						
						check = check_source_target_existance(VPUID, NAME, MODEL_NAME)
						if not check:
								continue
												
						#if check_sensitivity_existance(VPUID, NAME, MODEL_NAME):
						#		logging.info(f"{VPUID} {NAME} sensitivity exists")
						#		continue
						
						#if check_final_calibration(VPUID, NAME, MODEL_NAME):
						#		continue
						
						if check_validation_existance(VPUID, NAME, MODEL_NAME):
								logging.info(f"{VPUID} {NAME} sensitivity exists")
								continue
						
						logging.info(f"{VPUID} {NAME} sent to calibration")
						wrapped_model_evaluator = get_wrapped_model_evaluator(NAME, MODEL_NAME, VPUID, BASE_PATH, False, False, True)
						p = Process(target=wrapped_model_evaluator)
						p.start()
						logging.info(f"#####################{VPUID} {NAME} sent to process waiting for 5 minutes")
						processes.append(p)

						# Ensure CPU usage is low before starting new processes
						while not is_cpu_usage_low():
								logging.info('###########################\nCPU usage is high, waiting for 5 minutes\n####################################')
								time.sleep(5 * 60)

				# Join all processes
				for p in processes:
						p.join()

				logging.info('All processes finished')

				# Sleep before checking for new files again
				logging.info('###########################\nSleeping for 5 minutes\n####################################')
				time.sleep(5 * 60)
				