import logging


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
from ModelProcessing.processing_program import ProcessingProgram



if __name__ == "__main__":


	MODEL_NAME = 'SWAT_MODEL_Web_Application'
	username = "vahidr32"
	NAME = "05536265"
	## make a config dictornary
	sensitivity_flag = False
	calibration_flag = True
	verification_flag = False

	config = {
		'LEVEL': 'huc12',
		'NAME': NAME,
		'MODEL_NAME': MODEL_NAME,
		'VPUID': find_VPUID(NAME),	
		'BASE_PATH': f"/data/SWATGenXApp/Users/{username}",
		'username': username,
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
	
	processor = ProcessingProgram(config)
	processor.SWATGenX_SCV()