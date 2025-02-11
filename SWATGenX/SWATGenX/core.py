import subprocess
import os
import sys
import shutil
from SWATGenX.generate_swatplus_rasters import generate_swatplus_rasters
from SWATGenX.NHD_SWATPlus_Extractor import writing_swatplus_cli_files
from SWATGenX.PRISM_extraction import extract_PRISM_parallel
from SWATGenX.configuration import check_configuration
from SWATGenX.generate_swatplus_shapes import generate_swatplus_shapes
from SWATGenX.model_precipitation_info import plot_annual_precipitation
from SWATGenX.runQSWATPlus import runQSWATPlus
from SWATGenX.run_swatplusEditor import run_swatplus_editor
from SWATGenX.SWATplus_streamflow import fetch_streamflow_for_watershed
from SWATGenX.NSRDB_SWATplus_extraction import NSRDB_extract
from SWATGenX.SWATGenXLogging import LoggerSetup


def SWATGenXCore_run(SWATGenXPaths, swatgenx_config):
	"""
	Runs the SWATGenX model for the specified configuration.

	Args:
		swatgenx_config (dict): Configuration settings for the SWATGenX command.
	"""
	swatgenx = SWATGenXCore(SWATGenXPaths, swatgenx_config)
	swatgenx.process()
# Compare this snippet from SWATGenX/SWATGenX/SWATGenXConfigPars.py:

class SWATGenXCore:
	def __init__(self, SWATGenXPaths, config):
		"""
		Initializes the SWATGenXCore class with the provided configuration.

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
		self.paths = SWATGenXPaths
		self.paths.exe_start_year = 2020
		self.paths.exe_end_year = 2021

		self.logger = self.setup_logger()
		self.logger.info(f"SWATGenXCore: Starting the SWATGenX model for {self.site_no}")
		self.logger.info(f"SWAT outlet: {self.paths.swatgenx_outlet_path}")
	def setup_logger(self):
		"""Sets up the logger for the SWATGenXCore."""
		logger = LoggerSetup(
			rewrite=False,
			verbose=True,
		)
		return logger.setup_logger("SWATGenXCommand")

	def extract_metereological_data(self):
		os.makedirs(self.paths.extracted_swat_prism_path, exist_ok=True)
		self.logger.info(f"paths.extracted_swat_prism_path: {self.paths.extracted_swat_prism_path}")
		prism_grid_path = os.path.join(self.paths.extracted_swat_prism_path, "PRISM_grid.shp")
		pcp_cli = os.path.join(self.paths.extracted_swat_prism_path, "pcp.cli")
		tmp_cli = os.path.join(self.paths.extracted_swat_prism_path, "tmp.cli")
		slr_cli_path = os.path.join(self.paths.extracted_swat_prism_path, "slr.cli")
		hmd_cli_path = os.path.join(self.paths.extracted_swat_prism_path, "hmd.cli")
		wnd_cli_path = os.path.join(self.paths.extracted_swat_prism_path, "wnd.cli")

		if not os.path.exists(prism_grid_path) or not os.path.exists(pcp_cli) or not os.path.exists(tmp_cli):
			self.extract_prism_data()

		if not os.path.exists(slr_cli_path) or not os.path.exists(hmd_cli_path) or not os.path.exists(wnd_cli_path):
			self.logger.info(f"Extracting NSRDB data for {self.NAME}")
			self.extract_nsrdb_data()


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
			runQSWATPlus(self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME, self.paths)
			hru2_path = self.paths.construct_path(
				self.paths.swatgenx_outlet_path,
				self.VPUID,
				self.LEVEL,
				self.NAME,
				self.MODEL_NAME,
				"Watershed",
				"Shapes",
				"hrus2.shp",
			)
			success_flag = False
			self.logger.info(f"hu2_path: {hru2_path}")
			if not os.path.exists(hru2_path):
				self.logger.error(f"HRU2 shapefile does not exist for {self.NAME} after running QSWAT+")
				return False
			self.logger.info(f"QSWAT+ processes are completed for {self.NAME}, {self.VPUID}")
			return True
		except Exception as e:
			self.logger.error(f"Error in running QSWAT+ for {self.NAME}: {e}")
			return False

	def extract_prism_data(self):
		"""Extracts PRISM data for the watershed."""
		extract_PRISM_parallel(self.paths, self.VPUID, self.LEVEL, self.NAME, self.list_of_huc12s)
		plot_annual_precipitation(self.VPUID, self.LEVEL, self.NAME)
		writing_swatplus_cli_files(self.paths, self.VPUID, self.LEVEL, self.NAME)
		self.logger.info(f"Model extraction completed for {self.NAME}")

	def extract_nsrdb_data(self):
		NSRDB_extract(self.paths, self.VPUID, self.LEVEL, self.NAME)
		self.logger.info(f"NSRDB extraction completed for {self.NAME}")

	def write_meta_file(self):
		"""Writes model input characteristics to a meta file."""
		meta_file_path = self.paths.construct_path(
			self.paths.swatgenx_outlet_path,
			self.VPUID,
			self.LEVEL,
			self.NAME,
			self.MODEL_NAME,
			"meta.txt",
		)
		if not os.path.exists(meta_file_path):
			with open(meta_file_path, "w") as f:
				self.write_meta_data(f)

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
		execution_checkout_path = os.path.join(
			self.paths.swatgenx_outlet_path,
			self.VPUID,
			self.LEVEL,
			self.NAME,
			self.MODEL_NAME,
			"Scenarios",
			"Default",
			"TxtInOut",
			"simulation.out",
		)
		sim_file_exists = os.path.exists(execution_checkout_path)
		state = False

		if sim_file_exists:
			with open(execution_checkout_path, "r") as f:
				lines = f.readlines()
				for line in lines:
					if "Execution successfully completed" in line:
						print(f"Model already exists and successfully executed for {self.VPUID}/{self.LEVEL}/{self.NAME}")
						state = True

		if sim_file_exists and not state:
			raise ValueError(f"Model already exists but did not execute successfully for {self.VPUID}/{self.LEVEL}/{self.NAME}")
		
		return state
	

	def path_setup(self):
		"""Sets up the paths for the SWATGenX model."""
		
		self.paths.extracted_swat_prism_path = self.paths.construct_path(
			self.paths.swatgenx_outlet_path, self.VPUID, self.LEVEL, self.NAME, "PRISM"
		)


		streamflow_shape = self.paths.construct_path(
			self.paths.swatgenx_outlet_path, self.VPUID, self.LEVEL, self.NAME, "streamflow_data", "stations.shp"
		)
		
		if not os.path.exists(streamflow_shape) and os.path.exists(self.paths.extracted_swat_prism_path):
			self.logger.error(
				f"The model extraction has failed for {self.NAME} before (PRISM extracted but streamflow shapefile does not exist)"
			)
		

	def process(self):
		"""Core function to run the SWATGenX model for a given watershed."""
		self.EPSG = check_configuration(self.VPUID, self.landuse_epoch)
		self.logger.info(f"SWATGenXCore: Beginning the extraction of the model for {self.site_no}")
		self.NAME = self.site_no
		if state := self.check_simulation_output():
			return None
		self.path_setup()
		self.list_of_huc12s = self.prepare_huc12s()
		

		generate_swatplus_shapes(self.paths,
			self.list_of_huc12s, self.VPUID, self.LEVEL, self.NAME, self.EPSG, self.MODEL_NAME
		)
		try:
			generate_swatplus_rasters(
				self.paths,
				self.VPUID,
				self.LEVEL,
				self.NAME,
				self.MODEL_NAME,
				self.landuse_product,
				self.landuse_epoch,
				self.ls_resolution,
				self.dem_resolution,
			)
			
			self.logger.info(f"Generated SWAT+ shapes and rasters for {self.NAME}, {self.VPUID}")
		except Exception as e:
			self.logger.error(f"Error in generating SWAT+ shapes and rasters for {self.NAME}: {e}")
			return None

		self.list_of_huc12s = [f"{huc12:012d}" for huc12 in self.list_of_huc12s]
		self.logger.info(f"list of requested huc12s: {self.list_of_huc12s}")
		self.logger.info(f"Runnig QSWAT+ for {self.NAME}")
		
		success_flag = self.run_qswat_plus()

		if not success_flag:
			self.logger.error(f"QSWAT+ processes failed for {self.NAME}")
			return None

		self.logger.info(f"QSWAT+ processes are completed for {self.NAME}, {self.VPUID}")
		self.extract_metereological_data()

		run_swatplus_editor(self.paths, self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME)

		self.write_meta_file()

		fetch_streamflow_for_watershed(self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME)

		self.check_simulation_output()