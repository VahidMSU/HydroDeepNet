import os
import time
from functools import partial
from multiprocessing import Process
import logging
from ModelProcessing.utils import is_cpu_usage_low
from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
from ModelProcessing.processing_program import ProcessingProgram
from ModelProcessing.ModelConfig import ModelConfig
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
def wrapped_model_processor(VPUID, LEVEL, NAME, MODEL_NAME, sensitivity_flag=False, calibration_flag=True, verification_flag=False):

	config = ModelConfig(
		NAME=NAME,
		MODEL_NAME=MODEL_NAME,
		VPUID=VPUID,
		LEVEL=LEVEL,
		sensitivity_flag=sensitivity_flag,
		calibration_flag=calibration_flag,
		verification_flag=verification_flag,
		BASE_PATH=SWATGenXPaths.swatgenx_outlet_path,
		START_YEAR=2006,
		END_YEAR=2020,
		nyskip=3,
		sen_total_evaluations=1000,
		sen_pool_size=120,
		num_levels=10,
		cal_pool_size=50,
		max_cal_iterations=75,
		termination_tolerance=15,
		epsilon=0.001,
		Ver_START_YEAR=1997,
		Ver_END_YEAR=2020,
		Ver_nyskip=3,
		range_reduction_flag=False,
		pet=1,
		cn=1,
		no_value=1e6,
		verification_samples=5,
	)
		
	SCV_args = ProcessingProgram(config)

	if sensitivity_flag:
		SCV_args.process_sensitivity_stage()
		
	if calibration_flag:
		SCV_args.process_calibration_stage()

	if verification_flag:
		SCV_args.process_verification_stage()

	



def check_final_calibration(VPUID, LEVEL, NAME, MODEL_NAME):
	path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/best_solution_{MODEL_NAME}.txt"
	if not os.path.exists(path):
		print(f"{VPUID}/{LEVEL}/{NAME} calibration file does not exist")
		return False
	with open(path, "r") as f:
		lines = f.readlines()
		for line in lines:
			if "Final best" in line:
				best_solution = float(line.split(": ")[1].split("\n")[0])
				if best_solution >100:
					print(f"{VPUID}/{LEVEL}/{NAME} calibration is failed with best solution {best_solution}")
					return False   
				else: 
					print(f"{VPUID}/{LEVEL}/{NAME} calibration has been done with best solution {best_solution}")
					return True
	print(f"{VPUID}/{LEVEL}/{NAME} calibration is not completed")
	return False

def check_sensitivity_existance(VPUID, LEVEL, NAME, MODEL_NAME):
	path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/morris_Si_{MODEL_NAME}.csv"
	return bool(os.path.exists(path))

def check_validation_existance(VPUID, LEVEL, NAME, MODEL_NAME):
	path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/validation_stage_0"
	if not os.path.exists(path):
		return False

	files = os.listdir(path)
	return len(files) != 0

if __name__ == "__main__":

	BASE_PATH = SWATGenXPaths.swatgenx_outlet_path

	# Check if the path exists
	if os.path.exists(BASE_PATH):
		logging.info('Path exists')
	else:
		raise ValueError('Path does not exist')
	
	VPUID = '0407'
	LEVEL = 'huc12'
	INPUT_PATH = f'{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}'
	MODEL_NAME = 'SWAT_MODEL'
	
	processes = []
	
	NAMES = os.listdir(INPUT_PATH)

	if 'log.txt' in NAMES:
		NAMES.remove('log.txt')
	
	for NAME in NAMES:

		if check_final_calibration(VPUID, LEVEL, NAME, MODEL_NAME):
			print(f"{VPUID}/{LEVEL}/{NAME} calibration exists")
			continue

		logging.info(f"{VPUID}/{LEVEL}/{NAME} sent to calibration")
		wrapped_model_processor = wrapped_model_processor(VPUID, LEVEL, NAME, MODEL_NAME)
		
		p = Process(target=wrapped_model_processor)
		p.start()
		logging.info(f"#####################{VPUID}/{LEVEL}/{NAME} sent to process waiting for 60 seconds")
		processes.append(p)
		time.sleep(60)

		# Ensure CPU usage is low before starting new processes
		while not is_cpu_usage_low(n_processes=210):
			logging.info('###########################\nCPU usage is high, waiting for 50 minutes\n####################################')
			time.sleep(120 * 60)
		if len(processes) == 15:
			for p in processes:
				p.join()
			processes = []
	# Join all processes
	for p in processes:
		p.join()

	logging.info('All processes finished')

