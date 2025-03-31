from ModelProcessing.core import process_SCV_SWATGenXModel
import os
import time
from functools import partial
from multiprocessing import Process
import logging
from ModelProcessing.utils import is_cpu_usage_low

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='/data/SWATGenXApp/codes/ModelProcessing/logs/log.txt')
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

from ModelProcessing.find_VPUID import find_VPUID

###################### Note ##############################
## this script is designed to run Michigan models ##
###################### Note ##############################
def get_wrapped_model_evaluator(username, NAME, MODEL_NAME, BASE_PATH, sensitivity_flag, calibration_flag, verification_flag):
	
	
	## make a config dictornary

	config = {
		'LEVEL': 'huc12',
		'NAME': NAME,
		'MODEL_NAME': MODEL_NAME,
		'VPUID': find_VPUID(NAME),	
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




if __name__ == "__main__":

	BASE_PATH = '/data/MyDataBase/'

	# Check if the path exists
	if os.path.exists(BASE_PATH):
			logging.info('Path exists')
	else:
			raise ValueError('Path does not exist')

	INPUT_PATH = '/data2/MyDataBase/SWATplus_by_VPUID/0000/huc12'
	OUTPUT_PATH = '/data/MyDataBase/SWATplus_by_VPUID'
	MODEL_NAME = 'SWAT_MODEL_Web_Application'
	username = "vahidr32"
	NAME = "05536265"

	wrapped_model_evaluator = get_wrapped_model_evaluator(username, NAME, MODEL_NAME, BASE_PATH, False, True, True)
	


