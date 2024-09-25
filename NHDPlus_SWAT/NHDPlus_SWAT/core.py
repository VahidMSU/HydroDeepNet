from NHDPlus_SWAT.Process_raster import generate_raster_files
from NHDPlus_SWAT.NHD_SWAT_fun import writing_swatplus_cli_files
from NHDPlus_SWAT.PRISM_extraction import extract_PRISM_parallel
from NHDPlus_SWAT.configuration import check_configuration
from NHDPlus_SWAT.write_swatplus_shapes import generate_swatplus_shapes
from NHDPlus_SWAT.model_precipitation_info import plot_annual_precipitation
from NHDPlus_SWAT.runQSWATPlus import runQSWATPlus
from NHDPlus_SWAT.run_swatplusEditor import run_swatplusEditor
from NHDPlus_SWAT.SWATplus_streamflow import fetch_streamflow_for_watershed
from NHDPlus_SWAT.extract_SWAT_NSRDB import NSRDB_extract
import os
import logging
import shutil
def setup_logging(NAME, MODEL_NAME):
	"""Set up the logging configuration"""
	path_to_write = f"/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/{NAME}_{MODEL_NAME}.log"
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=path_to_write)
	### make sure it will write to the file
	logging.info(f"Logging setup for {NAME}")


def SWATGenXCore(site_no, BASE_PATH, VPUID, LEVEL, landuse_product, landuse_epoch, ls_resolution, dem_resolution, list_of_huc12s, MODEL_NAME):
	"""Core function to run the SWATGenX model for a given watershed"""
	setup_logging(site_no, MODEL_NAME)
	EPSG = check_configuration(VPUID, landuse_epoch)
	logging.info(f"SWATGenXCore: Beginning the extraction of the model for {site_no}")
	NAME = site_no
	SWAT_MODEL_PRISM_path = f'/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/PRISM/'

	#if MODEL_NAME in os.listdir(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}"):
	#	logging.info(f"Model already exists for {NAME} in SWAT_input")
#		return
	streamflow_shape = "/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/streamflow_data/stations.shp"
	if not os.path.exists(streamflow_shape) and os.path.exists(SWAT_MODEL_PRISM_path):
		logging.error(f"The model extraction has failed for {NAME} before (PRISM extracted but streamflow shapefile does not exist)")
	list_of_huc12s = list_of_huc12s[1:-1].split(", ")
	list_of_huc12s = {int(huc12.strip("'")) for huc12 in list_of_huc12s}  # Convert for loop into set comprehension

	generate_swatplus_shapes(list_of_huc12s, BASE_PATH, VPUID, LEVEL, NAME, EPSG, MODEL_NAME)  ## generating SWAT+ shapes
	generate_raster_files(BASE_PATH, VPUID, NAME, LEVEL, MODEL_NAME, landuse_product, landuse_epoch, ls_resolution, dem_resolution)                                 ### generating raster files
	logging.info(f"First Stage completed for {NAME}, {VPUID}")
	list_of_huc12s = [f'{huc12:012d}' for huc12 in list_of_huc12s]  ##
	print("list_of_huc12s:", list_of_huc12s)

	SWAT_MODEL_PRISM_output = os.path.join(SWAT_MODEL_PRISM_path, "PRISM_grid.shp")
	SWAT_MODEL_NSRDB_output = os.path.join(SWAT_MODEL_PRISM_path, "slr.cli")
	## we need to run a batch file to create the SWAT+ model
	try:
		runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME)
		hru2_path = f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{MODEL_NAME}/Watershed/Shapes/hrus2.shp"
		if not os.path.exists(hru2_path):
			logging.error(f"HRU2 shapefile does not exist for {NAME} after running QSWAT+")
			return None
		logging.info(f"QSWAT+ processes are completed for {NAME}, {VPUID}")
	except Exception as e:
		logging.error(f"Error in running QSWAT+ for {NAME}: {e}")
		return None
	## write PRISM data for the watershed
	if not os.path.exists(SWAT_MODEL_PRISM_output):
		try:
			extract_PRISM_parallel(VPUID, list_of_huc12s,SWAT_MODEL_PRISM_path)
			plot_annual_precipitation(BASE_PATH, VPUID, LEVEL, NAME)
			writing_swatplus_cli_files(BASE_PATH, VPUID, LEVEL, NAME)
			logging.info(f"Model extraction completed for {NAME}")
			NSRDB_extract(VPUID,NAME,LEVEL)
			logging.info(f"NSRDB extraction completed for {NAME}")
		except Exception as e:
			logging.error(f"Error in extracting PRISM data for {NAME}: {e}")
			return None
	run_swatplusEditor(VPUID, LEVEL, NAME, MODEL_NAME)

	execution_checkout_path =  f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/Default/TxtInOut/simulation.out"
	sim_file_exists = False
	execited_successfully = False

	fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MODEL_NAME)  ### fetching streamflow data for the watershed

	#### write model input characteristics in a meta file in the model project folder
	if not os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/meta.txt"):
		with open(f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/meta.txt", "w") as f:
			f.write(f"VPUID: {VPUID}\n")
			f.write(f"LEVEL: {LEVEL}\n")
			f.write(f"NAME: {NAME}\n")
			f.write(f"MODEL_NAME: {MODEL_NAME}\n")
			f.write(f"landuse_product: {landuse_product}\n")
			f.write(f"landuse_epoch: {landuse_epoch}\n")
			f.write(f"ls_resolution: {ls_resolution}\n")
			f.write(f"dem_resolution: {dem_resolution}\n")

	### also save it in the current directory
	if not os.path.exists("generated_models/"):
		os.makedirs("generated_models/")
	if not os.path.exists("generated_models/logs.txt"):
		with open("generated_models/logs.txt", "w") as f:
			f.write(f"VPUID: {VPUID}\n")
			f.write(f"LEVEL: {LEVEL}\n")
			f.write(f"NAME: {NAME}\n")
			f.write(f"MODEL_NAME: {MODEL_NAME}\n")
			f.write(f"landuse_product: {landuse_product}\n")
			f.write(f"landuse_epoch: {landuse_epoch}\n")
			f.write(f"ls_resolution: {ls_resolution}\n")
			f.write(f"dem_resolution: {dem_resolution}\n")



	if os.path.exists(execution_checkout_path):
		sim_file_exists = True
		execited_successfully = False
		with open(execution_checkout_path, "r") as f:
			lines = f.readlines()
			for line in lines:
				if "Execution successfully completed" in line:
					print(f"Model already exists and successfully executed for {NAME}")

					execited_successfully = True
					### copy to the SWAT_input if VPUID is among [0450, 0406]
					if VPUID in ['0405', '0406', '0407', '0408', '0410']:
						if not os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}"):
							shutil.copytree(f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}", f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}")

							logging.info(f"Model successfully copied to SWAT_input for {NAME}")
						else:
							logging.info(f"Model already exists for {NAME} in SWAT_input")
						if not os.path.exists(f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{MODEL_NAME}"):
							shutil.copytree(f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}", f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/{MODEL_NAME}")
							logging.info(f"Model successfully copied to SWAT_input for {NAME}")
						else:
							logging.info(f"Model already exists for {NAME} in SWAT_input")

					return

	if sim_file_exists and not execited_successfully:
		raise ValueError(f"Model already exists but did not execute successfully for {NAME}")
	# get the streamflow data for the watershed