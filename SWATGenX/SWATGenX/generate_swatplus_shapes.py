import os
import pandas as pd
import time
from SWATGenX.NHD_SWATPlus_Extractor import NHD_SWATPlus_Extractor


def generate_swatplus_shapes(SWATGenXPaths, list_of_huc12s, VPUID, LEVEL, NAME, EPSG, MODEL_NAME):

	SWAT_MODEL_directory = f'{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/'  # Move assignment closer to its usage within a block

	output = f'{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}'
	os.makedirs(output, exist_ok=True)
	
	if os.path.exists(SWAT_MODEL_directory):
		print(f"SWAT+ The directory already exists {SWAT_MODEL_directory}")
	else:
		os.makedirs(SWAT_MODEL_directory, exist_ok=True)

	print("SWAT_MODEL_directory:", SWAT_MODEL_directory)
	print("list_of_huc12s:", list_of_huc12s)

	### make it a list from a set
	list_of_huc12s = list(list_of_huc12s)
	## get the type of list_of_huc12s

	if isinstance(list_of_huc12s[0], str):
		print("list_of_huc12s is a list of strings")
		list_of_huc12s = [int(huc12) for huc12 in list_of_huc12s]

	extractor = NHD_SWATPlus_Extractor(SWATGenXPaths, list_of_huc12s, LEVEL, VPUID, MODEL_NAME, NAME)
	
	streams = extractor.extract_initial_streams()

	streams = extractor.incorporate_lakes(streams)  # Remove unused variable "Lakes"

	streams = extractor.include_lakes_in_streams(streams)

	extractor.write_output(streams, EPSG)

	extractor.creating_modified_inputs()

	print(f"SWATPlus shapes are generated for {NAME}")

