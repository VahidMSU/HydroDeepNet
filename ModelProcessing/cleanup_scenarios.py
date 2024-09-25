import os
import time
import shutil
from ModelProcessing.utils import copy_files
import logging

VPUID = '0000'
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12")
NAMES.remove("log.txt")
INPUT_PATH = '/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12'
MODEL_NAME = "SWAT_gwflow_MODEL"
OUTPUT_PATH = '/data/MyDataBase/SWATplus_by_VPUID'


for NAME in NAMES:
	path_to_copy = {
		'model': os.path.join(INPUT_PATH, NAME, MODEL_NAME),
		'streamflow': os.path.join(INPUT_PATH, NAME, 'streamflow_data'),
		'best': os.path.join(INPUT_PATH, NAME, f'best_solution_{MODEL_NAME}.txt'),
		'local': os.path.join(INPUT_PATH, NAME, f'local_best_solution_{MODEL_NAME}.txt'),
		'cal': os.path.join(INPUT_PATH, NAME, f'cal_parms_{MODEL_NAME}.cal')
	}




	if not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/"):

		copy_files(NAME, VPUID, MODEL_NAME, path_to_copy, OUTPUT_PATH)

	path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/"

	if os.path.exists(path):
		# Remove all directories starting with Scenario_
		for scenario in os.listdir(path):
			if scenario.startswith("Scenario_"):
				shutil.rmtree(os.path.join(path, scenario))

	if not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/morris_Si_{MODEL_NAME}.csv"):

		if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/sensitivity_performance_scores.txt"):
			os.remove(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/sensitivity_performance_scores.txt")

	if not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/log.txt"):
		with open(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/log.txt", "w") as f:
			f.write("")

	if not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default"):
		print(f"Default scenario does not exist for {NAME}")
	else:
		## check simulation.out for Execution successfully completed
		with open(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/simulation.out", "r") as f:
			lines = f.readlines()
			flag = any("Execution successfully completed" in line for line in lines)
			if not flag:
				print(f"Execution failed for {NAME}")
	files = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/")
	### remove all files ending with txt
	for file in files:
		if file.endswith(".txt"):
			os.remove(os.path.join(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/", file))
	if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/best_solution_{MODEL_NAME}.txt"):
		with open(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/best_solution_{MODEL_NAME}.txt", "r") as f:
			lines = f.readlines()
			final_best = False
			for line in lines:
				if "Final best objective value" in line:
					final_best = True
					break
			if not final_best:
				print(f"Final best objective value not found for {NAME}")
				## delete the directories: figures_{MODEL_NAME}_calibration_daily, figures_{MODEL_NAME}_calibration_monthly, calibration_figures_{MODEL_NAME},calibration_performance_scores.txt, local_best_solution_{MODEL_NAME}.txt, best_solution_{MODEL_NAME}.txt, initial_point_calibration_{MODEL_NAME}.txt
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_{MODEL_NAME}_calibration_daily"):
					shutil.rmtree(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_{MODEL_NAME}_calibration_daily")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_{MODEL_NAME}_calibration_monthly"):
					shutil.rmtree(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_{MODEL_NAME}_calibration_monthly")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/calibration_figures_{MODEL_NAME}"):
					shutil.rmtree(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/calibration_figures_{MODEL_NAME}")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/calibration_performance_scores.txt"):
					os.remove(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/calibration_performance_scores.txt")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/local_best_solution_{MODEL_NAME}.txt"):
					os.remove(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/local_best_solution_{MODEL_NAME}.txt")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/best_solution_{MODEL_NAME}.txt"):
					os.remove(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/best_solution_{MODEL_NAME}.txt")
				if os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/initial_point_calibration_{MODEL_NAME}.txt"):
					os.remove(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/initial_point_calibration_{MODEL_NAME}.txt")


				continue