import os
import pandas as pd
import time
from NHDPlus_SWAT.NHD_SWAT_fun import start_extracting, incorporate_lakes, include_lakes_in_streams, write_output, creating_modified_inputs

def generate_swatplus_shapes(list_of_huc12s, BASE_PATH, VPUID, LEVEL, NAME, EPSG, MODEL_NAME):

	SWAT_MODEL_directory = os.path.join(BASE_PATH, 'SWATplus_by_VPUID', f'{VPUID}/{LEVEL}/{NAME}/')  # Move assignment closer to its usage within a block

	directory = os.path.join(BASE_PATH, 'SWATplus_by_VPUID', f'{VPUID}/{LEVEL}/{NAME}')
	os.makedirs(directory, exist_ok=True)
	if os.path.exists(SWAT_MODEL_directory):
		print(f"SWAT+ The directory already exists {SWAT_MODEL_directory}")
	else:
		os.makedirs(SWAT_MODEL_directory, exist_ok=True)

	print("SWAT_MODEL_directory:", SWAT_MODEL_directory)
	list_of_huc12s = list(list_of_huc12s)
	print("list_of_huc12s:", list_of_huc12s)
	## get the type of list_of_huc12s
	if isinstance(list_of_huc12s[0], str):
		print("list_of_huc12s is a list of strings")
		list_of_huc12s = [int(huc12) for huc12 in list_of_huc12s]

	streams = start_extracting(BASE_PATH, list_of_huc12s, LEVEL, VPUID)

	### test if the streams are dataframe, otherwise exit

	if not isinstance(streams, pd.DataFrame):
		print("The streams are not a dataframe")
		time.sleep(100)
	else:
		print("The streams are a dataframe")
	streams = incorporate_lakes(BASE_PATH, streams, VPUID)  # Remove unused variable "Lakes"

	if not isinstance(streams, pd.DataFrame):
		print("The streams are not a dataframe")
		time.sleep(100)
	else:
		print("The streams are a dataframe")
	streams = include_lakes_in_streams(streams)

	write_output(BASE_PATH, streams, LEVEL, NAME, VPUID, EPSG, MODEL_NAME)

	creating_modified_inputs(BASE_PATH, VPUID, LEVEL, NAME, MODEL_NAME)

	print(f"SWATPlus shapes are generated for {NAME}")
