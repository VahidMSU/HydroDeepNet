import sys
import os
import json
import logging
import argparse
from ModelProcessing.processing_program import ProcessingProgram
from ModelProcessing.config import ModelConfig
from ModelProcessing.logging_utils import setup_logger, get_logger
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

def parse_args():
	parser = argparse.ArgumentParser(description='Run ModelProcessing script with configurations.')
	parser.add_argument('--config', type=str, help='Path to JSON config file')
	parser.add_argument('--username', type=str, help='Username', required=True)
	parser.add_argument('--vpuid', type=str, help='VPUID')
	parser.add_argument('--level', type=str, default='huc12', help='Level (default: huc12)')
	parser.add_argument('--name', type=str, help='Name')
	parser.add_argument('--model-name', type=str, help='Model name', required=True)
	parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis', default=False)
	parser.add_argument('--calibration', action='store_true', help='Run calibration', default=True)
	parser.add_argument('--verification', action='store_true', help='Run verification', default=False)
	
	return parser.parse_args()

def main():
	# Initialize the central application logger
	logger = setup_logger(
		name='ModelProcessing',
		log_file='/data/SWATGenXApp/codes/ModelProcessing/logs/ModelProcessing.log',
		level=logging.INFO
	)
	
	logger.info("Starting ModelProcessing application")
	
	args = parse_args()
	
	# Load configuration from file if provided
	if args.config:
		logger.info(f"Loading configuration from {args.config}")
		try:
			with open(args.config, 'r') as f:
				config_data = json.load(f)
			config = ModelConfig(**config_data)
			logger.info("Configuration loaded successfully")
		except Exception as e:
			logger.error(f"Failed to load configuration from {args.config}: {str(e)}")
			sys.exit(1)
	else:
		# Auto-determine VPUID from name if not provided
		vpuid = args.vpuid
		if not vpuid and args.name:
			logger.info(f"VPUID not provided, determining from NAME: {args.name}")
			# Use the station ID/name as the VPUID when it's not explicitly provided
			from ModelProcessing.find_VPUID import find_VPUID
			vpuid = find_VPUID(args.name)
			logger.info(f"Auto-determined VPUID: {vpuid}")
		
		# Use command line arguments to create config
		logger.info("Creating configuration from command line arguments")
		config = ModelConfig(
			username=args.username,
			VPUID=vpuid,
			LEVEL=args.level,
			NAME=args.name,
			MODEL_NAME=args.model_name,
			sensitivity_flag=args.sensitivity,
			calibration_flag=args.calibration,
			verification_flag=args.verification
		)
	
	# Log the configuration
	logger.info(f"Configuration: Model={config.MODEL_NAME}, Name={config.NAME}, VPUID={config.VPUID}")
	logger.info(f"Process flags: Sensitivity={config.sensitivity_flag}, Calibration={config.calibration_flag}, Verification={config.verification_flag}")
	
	# Run the processing program
	try:
		process = ProcessingProgram(config)
		result = process.SWATGenX_SCV()
		logger.info(f"Processing completed with result: {result}")
	except Exception as e:
		logger.error(f"Processing failed: {str(e)}", exc_info=True)
		sys.exit(1)
	
	logger.info("ModelProcessing application completed successfully")
	sys.exit(0)

if __name__ == "__main__":
	main()