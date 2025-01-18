from SWATGenX.core import SWATGenXCore
import geopandas as gpd
from SWATGenX.SWATGenXLogging import LoggerSetup
import os
import pandas as pd
from functools import partial
from multiprocessing import Process
#from app.models import User  # Assuming User is defined in app.models
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import get_all_VPUIDs
except Exception:
    from SWATGenXConfigPars import SWATGenXPaths
    from utils import get_all_VPUIDs
try:
	from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
	from SWATGenXConfigPars import SWATGenXPaths

class SWATGenXCommand:
	def __init__(self, swatgenx_config):
		"""
		Initializes the SWATGenXCommand class with the provided configuration.

		This class handles the extraction and processing of hydrological data based on the specified configuration. It supports different levels of data processing, including HUC12, HUC4, and HUC8.

		Args:
			swatgenx_config (dict): Configuration settings for the SWATGenX command.
		"""
		self.config = swatgenx_config
		self.logger = LoggerSetup(verbose=True, rewrite=True).setup_logger("SWATGenXCommand")

	def find_VPUID(self, station_no, level="huc12"):
		"""
		Finds the VPUID for a given station number.

		This method retrieves the VPUID based on the specified level. It reads from a CSV file containing streamflow station data.

		Args:
			station_no (str): The station number to find the VPUID for.
			level (str): The level of detail for the VPUID (default is "huc12").

		Returns:
			str: The corresponding VPUID.
		"""
		if level == "huc8":
			return f"0{int(station_no)[:3]}"

		conus_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str, 'huc_cd': str})
		return conus_streamflow_data[conus_streamflow_data.site_no == station_no].huc_cd.values[0][:4]
	def generate_huc12_list(self, huc8, vpuid):
		"""
		Generates a list of HUC12s for a given HUC8 and VPUID.

		This method reads the relevant geospatial data from a File Geodatabase and intersects HUC12 with HUC8 to extract the corresponding HUC12s.

		Args:
			huc8 (str): The HUC8 code to filter by.
			vpuid (str): The VPUID to use for data retrieval.

		Returns:
			dict: A dictionary mapping HUC8 codes to their corresponding HUC12 values.
		"""
		path = f"{SWATGenXPaths.extracted_nhd_swatplus_path}/{vpuid}/unzipped_NHDPlusVPU/"
		gdb_files = [g for g in os.listdir(path) if g.endswith('.gdb')]
		
		if not gdb_files:
			raise FileNotFoundError(f"No .gdb files found in the directory: {path}")

		path = os.path.join(path, gdb_files[0])
		huc12 = gpd.read_file(path, driver='FileGDB', layer='WBDHU12').to_crs('EPSG:4326')
		huc8_layer = gpd.read_file(path, driver='FileGDB', layer='WBDHU8').to_crs('EPSG:4326')

		# Intersect HUC8 with HUC12
		huc12 = gpd.overlay(huc12, huc8_layer, how='intersection')

		if "huc8" in huc12.columns:
			huc12.rename(columns={"huc8": "HUC8"}, inplace=True)
			huc12['HUC8'] = huc12['HUC8'].astype(str).str.zfill(8)

		# Filter HUC12 based on the provided HUC8
		huc12_filtered = huc12[huc12['HUC8'] == huc8]

		if "huc12" in huc12_filtered.columns:
			huc12_filtered.rename(columns={"huc12": "HUC12"}, inplace=True)

		if huc12_filtered.empty:
			self.logger.warning(f"No HUC12s found for HUC8: {huc8} and VPUID: {vpuid}")
			return {}

		huc12_grouped = huc12_filtered.groupby('HUC8')
		return {huc8: group['HUC12'].values for huc8, group in huc12_grouped}

	def return_list_of_huc12s(self, station_name, max_area):
		"""
		Returns a list of HUC12s for a given station name and maximum drainage area.

		This method checks the drainage area of the specified station and retrieves eligible HUC12s based on the maximum area criteria.

		Args:
			station_name (str): The name of the station to retrieve HUC12s for.
			max_area (float): The maximum drainage area allowed.

		Returns:
			tuple: A tuple containing the list of HUC12s and the corresponding VPUID.
		"""
		vpuid = self.find_VPUID(station_name)
		streamflow_metadata = f"{SWATGenXPaths.streamflow_path}/VPUID/{vpuid}/meta_{vpuid}.csv"
		streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})

		drainage_area = streamflow_metadata[streamflow_metadata.site_no == station_name].drainage_area_sqkm.values[0]

		eligible_stations = streamflow_metadata[streamflow_metadata.drainage_area_sqkm < max_area]
		if len(eligible_stations) == 0:
			self.logger.error(f"Station {station_name} does not meet the maximum drainage area criteria: {max_area} sqkm")
			return None

		if len(eligible_stations[eligible_stations.site_no == station_name]) == 0:
			self.logger.error(f"Station {drainage_area} sqkm is greater than the maximum drainage area criteria: {max_area} sqkm")
			return None

		list_of_huc12s = eligible_stations[eligible_stations.site_no == station_name].list_of_huc12s.values[0]
		print(f"Station name: {station_name}, VPUID: {vpuid}")

		return list_of_huc12s, vpuid
	def execute(self):
		"""
		Executes the SWATGenX command based on the provided configuration.

		This method processes the configuration to determine the level of data extraction and initiates the appropriate data handling procedures.

		Returns:
			str: The path to the processed data output.
		"""
		level = self.config.get("LEVEL")

		if level == "huc12":
			return self.handle_huc12()

		elif level == "huc4":
			return self.handle_huc4()

		elif level == "huc8":
			return self.handle_huc8()

	def handle_huc12(self):
		"""Handles the HUC12 level processing."""
		self.logger.info(f'LEVEL: huc12, station_name: {self.config.get("station_name")}')
		station_name = self.config.get("station_name")
		list_of_huc12s, vpuid = self.return_list_of_huc12s(station_name, self.config.get("MAX_AREA"))

		# Update the configuration dictionary
		self.config.update({
			"site_no": station_name,
			"VPUID": vpuid,
			"LEVEL": "huc12",
			"list_of_huc12s": list_of_huc12s,
		})
		core = SWATGenXCore(self.config)
		core.process()
		return os.path.join(SWATGenXPaths.database_dir, f"SWATplus_by_VPUID/{vpuid}/huc12/{station_name}/")

	def handle_huc4(self):
		"""Handles the HUC4 level processing."""
		vpuid_list = get_all_VPUIDs()
		processes = []

		for vpuid in vpuid_list:
			if vpuid[:2] != self.config.get("region"):
				continue
			eligible_stations = self.get_eligible_stations(vpuid)

			if len(eligible_stations) == 0:
				self.logger.error(f"No eligible stations found for VPUID: {vpuid}")
				return None

			for site_no in eligible_stations.site_no:
				self.process_station(site_no, vpuid, eligible_stations)

		for p in processes:
			p.join()

	def handle_huc8(self):
		"""Handles the HUC8 level processing."""
		vpuid, huc8_name = self.extract_vpuid_and_huc8_name()
		list_of_huc12s = self.generate_huc12_list(huc8_name, vpuid)
		self.config.update({
			"site_no": huc8_name,
			"VPUID": vpuid,
			"LEVEL": "huc8",
			"list_of_huc12s": list_of_huc12s,
		})

		core = SWATGenXCore(self.config)
		core.process()

		return os.path.join(SWATGenXPaths.database_dir, f"SWATplus_by_VPUID/{vpuid}/huc8/{huc8_name}/")

	def get_eligible_stations(self, vpuid):
		"""Retrieves eligible stations based on drainage area criteria."""
		streamflow_metadata = f"{SWATGenXPaths.streamflow_path}/VPUID/{vpuid}/meta_{vpuid}.csv"
		streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})
		eligible_stations = streamflow_metadata[
			(streamflow_metadata.drainage_area_sqkm < self.config.get("MAX_AREA")) &
			(streamflow_metadata.drainage_area_sqkm > self.config.get("MIN_AREA"))
		]
		eligible_stations = eligible_stations[eligible_stations.GAP_percent < self.config.get("GAP_percent")]
		return eligible_stations

	def process_station(self, site_no, vpuid, eligible_stations):
		"""Processes a single station by updating the configuration and starting a new process."""
		print(f"site_no: {site_no}")
		self.config.update({
			"site_no": site_no,
			"VPUID": vpuid,
			"list_of_huc12s": eligible_stations[eligible_stations.site_no == site_no].list_of_huc12s.values[0],
		})
		wrapped_SWATGenXCore = partial(SWATGenXCore, self.config)

		p = Process(target=wrapped_SWATGenXCore)
		p.start()
		return p

	def extract_vpuid_and_huc8_name(self):
		"""
		Extracts the VPUID and HUC8 name from the configuration.

		This method retrieves the VPUID and HUC8 name based on the station name provided in the configuration.

		Returns:
			tuple: A tuple containing the VPUID and HUC8 name.
		"""
		vpuid = self.config.get("station_name")[:4]
		huc8_name = self.config.get("station_name")
		return vpuid, huc8_name
