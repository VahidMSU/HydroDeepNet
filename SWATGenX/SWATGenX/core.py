import os
import shutil
import pandas as pd
from SWATGenX.generate_swatplus_rasters import generate_swatplus_rasters
from SWATGenX.NHD_SWATPlus_Extractor import writing_swatplus_cli_files
from SWATGenX.PRISM_extraction import extract_PRISM_parallel
from SWATGenX.configuration import check_configuration
from SWATGenX.generate_swatplus_shapes import generate_swatplus_shapes
from SWATGenX.model_precipitation_info import plot_annual_precipitation
from SWATGenX.runQSWATPlus import runQSWATPlus
from SWATGenX.run_swatplusEditor import run_swatplusEditor
from SWATGenX.SWATplus_streamflow import fetch_streamflow_for_watershed
from SWATGenX.NSRDB_SWATplus_extraction import NSRDB_extract
from SWATGenX.SWATGenXLogging import LoggerSetup
from functools import partial

try:
	from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
	from SWATGenXConfigPars import SWATGenXPaths

class SWATGenXCore:
	def __init__(self, config):
		"""
		Initializes the SWATGenXCore class with the provided configuration.

		This class handles the core operations for running the SWATGenX model, including data extraction, processing, and logging.

		Args:
			config (dict): Configuration settings for the SWATGenX command.
		"""
		
		self.config = config
		self.site_no = config.get("site_no")
		self.VPUID = config.get("VPUID")
		self.LEVEL = config.get("LEVEL")
		self.landuse_product = config.get("landuse_product")
		self.landuse_epoch = config.get("landuse_epoch")
		self.ls_resolution = config.get("ls_resolution")
		self.dem_resolution = config.get("dem_resolution")
		self.list_of_huc12s = config.get("list_of_huc12s")
		self.MODEL_NAME = config.get("MODEL_NAME")
		self.logger = self.setup_logger()
		self.logger.info(f"SWATGenXCore: Starting the SWATGenX model for {self.site_no}")

	def setup_logger(self):
		"""Sets up the logger for the SWATGenXCore."""
		logger = LoggerSetup(
			rewrite=False,
			verbose=True,
		)
		return logger.setup_logger("SWATGenXCommand")

	def process(self):
		"""Core function to run the SWATGenX model for a given watershed."""
		self.EPSG = check_configuration(self.VPUID, self.landuse_epoch)
		self.logger.info(f"SWATGenXCore: Beginning the extraction of the model for {self.site_no}")
		self.NAME = self.site_no
		self.SWAT_MODEL_PRISM_path = f'{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/PRISM/'

		streamflow_shape = f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/streamflow_data/stations.shp"

		if not os.path.exists(streamflow_shape) and os.path.exists(self.SWAT_MODEL_PRISM_path):
			self.logger.error(f"The model extraction has failed for {self.NAME} before (PRISM extracted but streamflow shapefile does not exist)")

		self.list_of_huc12s = self.prepare_huc12s()

		generate_swatplus_shapes(self.list_of_huc12s, self.VPUID, self.LEVEL, self.NAME, self.EPSG, self.MODEL_NAME)
		generate_swatplus_rasters(self.VPUID, self.NAME, self.LEVEL, self.MODEL_NAME, self.landuse_product, self.landuse_epoch, self.ls_resolution, self.dem_resolution)
		self.logger.info(f"First Stage completed for {self.NAME}, {self.VPUID}")

		self.list_of_huc12s = [f'{huc12:012d}' for huc12 in self.list_of_huc12s]
		print("list_of_huc12s:", self.list_of_huc12s)

		self.run_qswat_plus()

		SWAT_MODEL_PRISM_output = os.path.join(self.SWAT_MODEL_PRISM_path, "PRISM_grid.shp")
		SWAT_MODEL_NSRDB_output = os.path.join(self.SWAT_MODEL_PRISM_path, "slr.cli")

		if not os.path.exists(SWAT_MODEL_PRISM_output):
			self.extract_prism_data()

		if not os.path.exists(SWAT_MODEL_NSRDB_output):
			self.extract_nsrdb_data()

		run_swatplusEditor(self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME)
		self.write_meta_file()

		self.check_simulation_output()

	def prepare_huc12s(self):
		"""Prepares the list of HUC12s based on the specified level."""
		if self.LEVEL in ["huc12", "huc4"]:
			return {int(huc12.strip("'")) for huc12 in self.list_of_huc12s[1:-1].split(", ")}

		elif self.LEVEL == "huc8":
			print("list_of_huc12s:", self.list_of_huc12s)
			return {int(huc12) for huc12 in self.list_of_huc12s[self.site_no]}

	def run_qswat_plus(self):
		"""Runs the QSWAT+ model for the specified site."""
		try:
			runQSWATPlus(self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME)
			hru2_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Watershed/Shapes/hrus2.shp"
			if not os.path.exists(hru2_path):
				self.logger.error(f"HRU2 shapefile does not exist for {self.NAME} after running QSWAT+")
				return None
			self.logger.info(f"QSWAT+ processes are completed for {self.NAME}, {self.VPUID}")
		except Exception as e:
			self.logger.error(f"Error in running QSWAT+ for {self.NAME}: {e}")
			return None

	def extract_prism_data(self):
		"""Extracts PRISM data for the watershed."""
		extract_PRISM_parallel(self.VPUID, self.list_of_huc12s, self.SWAT_MODEL_PRISM_path)
		plot_annual_precipitation(SWATGenXPaths.database_dir, self.VPUID, self.LEVEL, self.NAME)
		writing_swatplus_cli_files(SWATGenXPaths.database_dir, self.VPUID, self.LEVEL, self.NAME)
		self.logger.info(f"Model extraction completed for {self.NAME}")

	def extract_nsrdb_data(self):
		NSRDB_extract(self.VPUID, self.NAME, self.LEVEL)
		self.logger.info(f"NSRDB extraction completed for {self.NAME}")

	def write_meta_file(self):
		"""Writes model input characteristics to a meta file."""
		meta_file_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/meta.txt"
		if not os.path.exists(meta_file_path):
			with open(meta_file_path, "w") as f:
				self.write_meta_data(f)
		# Save to the current directory
		if not os.path.exists("generated_models/"):
			os.makedirs("generated_models/")
		if not os.path.exists("generated_models/logs.txt"):
			with open("generated_models/logs.txt", "w") as f:
				self.write_meta_data(f)

	def write_meta_data(self, f):
		f.write(f"VPUID: {self.VPUID}\n")
		f.write(f"LEVEL: {self.LEVEL}\n")
		f.write(f"self.NAME: {self.NAME}\n")
		f.write(f"MODEL_NAME: {self.MODEL_NAME}\n")
		f.write(f"landuse_product: {self.landuse_product}\n")
		f.write(f"landuse_epoch: {self.landuse_epoch}\n")
		f.write(f"ls_resolution: {self.ls_resolution}\n")
		f.write(f"dem_resolution: {self.dem_resolution}\n")

	def check_simulation_output(self):
		"""Checks the simulation output for successful execution."""
		execution_checkout_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/Scenarios/Default/TxtInOut/simulation.out"
		sim_file_exists = os.path.exists(execution_checkout_path)
		excited_successfully = False

		if sim_file_exists:
			with open(execution_checkout_path, "r") as f:
				lines = f.readlines()
				for line in lines:
					if "Execution successfully completed" in line:
						print(f"Model already exists and successfully executed for {self.NAME}")
						excited_successfully = True
						self.copy_to_swat_input()

		if sim_file_exists and not excited_successfully:
			raise ValueError(f"Model already exists but did not execute successfully for {self.NAME}")

	def copy_to_swat_input(self):
		"""Copies the model to the SWAT_input directory if applicable."""
		if self.VPUID in ['0405', '0406', '0407', '0408', '0410']:
			if not os.path.exists(f"{SWATGenXPaths.database_dir}/SWAT_input/huc12/{self.NAME}"):
				shutil.copytree(f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}", f"{SWATGenXPaths.database_dir}/SWAT_input/huc12/{self.NAME}")
				self.logger.info(f"Model successfully copied to SWAT_input for {self.NAME}")
			else:
				self.logger.info(f"Model already exists for {self.NAME} in SWAT_input")

			if not os.path.exists(f"{SWATGenXPaths.database_dir}/SWAT_input/huc12/{self.NAME}/{self.MODEL_NAME}"):
				shutil.copytree(f"{SWATGenXPaths.swatgenx_outlet_path}/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}", f"{SWATGenXPaths.database_dir}/SWAT_input/huc12/{self.NAME}/{self.MODEL_NAME}")
				self.logger.info(f"Model successfully copied to SWAT_input for {self.NAME}")
			else:
				self.logger.info(f"Model already exists for {self.NAME} in SWAT_input")