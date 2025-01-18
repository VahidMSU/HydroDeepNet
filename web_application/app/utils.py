import sys
sys.path.append(r'/data/SWATGenXApp/codes/SWATGenX')
from SWATGenX.SWATGenXCommand import SWATGenXCommand
from SWATGenX.integrate_streamflow_data import integrate_streamflow_data
from SWATGenX.find_station_region import find_station_region
#sys.path.append(r'/data/SWATGenXApp/codes/ModelProcessing') ## not yet created
#from ModelProcessing.core import process_SCV_SWATGenXModel
import os
import shutil
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import logging
from shapely.geometry import mapping
from scipy.spatial import cKDTree


class LoggerSetup:
	def __init__(self, report_path, verbose=True, rewrite=False):
		"""
		Initialize the LoggerSetup class.

		Args:
			report_path (str): Path to the directory where the log file will be saved.
			verbose (bool): Whether to print logs to console. Defaults to True.
		"""
		self.report_path = report_path
		self.logger = None
		self.verbose = verbose
		self.rewrite = rewrite  

	def setup_logger(self, name="GeoClassCNNLogger"):
		"""
		Set up the logger to log messages to a file and optionally to the console.

		Returns:
			logging.Logger: Configured logger.
		"""
		if not self.logger:
			# Define the path for the log file
			path = os.path.join(self.report_path, f"{name}.log")
			if self.rewrite and os.path.exists(path):
				os.remove(path)
			# Create a logger
			self.logger = logging.getLogger(name)
			self.logger.setLevel(logging.INFO)  # Set the logging level

			# FileHandler for logging to a file
			file_handler = logging.FileHandler(path)
			file_handler.setLevel(logging.INFO)
			self.logger.addHandler(file_handler)

			# Conditionally add console handler based on verbose flag
			if self.verbose:
				console_handler = logging.StreamHandler()
				console_handler.setLevel(logging.INFO)
				self.logger.addHandler(console_handler)

			self.logger.info(f"Logging to {path}")

		return self.logger
	def error(self, message, time_stamp=True):
		"""
		Log an error message.

		Args:
			message (str): The error message to log.
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		self.info(message, level="error", time_stamp=time_stamp)

	def warning(self, message, time_stamp=True):
		"""
		Log a warning message.

		Args:
			message (str): The warning message to log.
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		self.info(message, level="warning", time_stamp=time_stamp)

	def info(self, message, level="info", time_stamp=True):
		"""
		Log a message with or without a timestamp.

		Args:
			message (str): The message to log.
			level (str): The logging level (e.g., "info", "error").
			time_stamp (bool): Whether to include a timestamp in the log.
		"""
		# Create a temporary logger with the desired format
		temp_logger = logging.getLogger("TempLogger")
		temp_logger.setLevel(self.logger.level)

		# Remove existing handlers to avoid duplicates
		temp_logger.handlers.clear()

		# Define the log format based on the time_stamp flag
		log_format = '%(asctime)s - %(levelname)s - %(message)s' if time_stamp else '%(levelname)s - %(message)s'

		# Add file handler
		for handler in self.logger.handlers:
			if isinstance(handler, logging.FileHandler):
				new_file_handler = logging.FileHandler(handler.baseFilename)
				new_file_handler.setFormatter(logging.Formatter(log_format))
				temp_logger.addHandler(new_file_handler)

		# Conditionally add console handler based on verbose flag
		if self.verbose:
			console_handler = logging.StreamHandler()
			console_handler.setFormatter(logging.Formatter(log_format))
			temp_logger.addHandler(console_handler)

		# Log the message at the specified level
		log_methods = {
			"info": temp_logger.info,
			"error": temp_logger.error,
			"warning": temp_logger.warning,
			"debug": temp_logger.debug
		}
		log_method = log_methods.get(level.lower(), temp_logger.info)
		log_method(message)

def hydrogeo_dataset_dict(path="/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"):
    with h5py.File(path, 'r') as f:
        groups = f.keys()
        hydrogeo_dict = {group: list(f[group].keys()) for group in groups}
    return hydrogeo_dict

def CDL_lookup(code):
    path = "/data/SWATGenXApp/GenXAppData/CDL/CDL_CODES.csv"
    df = pd.read_csv(path)
    df = df[df['CODE'] == code]
    return df.NAME.values[0]

def read_h5_file(address, lat=None, lon=None, lat_range=None, lon_range=None, logger=None):
    path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"

    if lat is not None and lon is not None:
        if logger:
            logger.info(f"requested lat and lon: {lat}, {lon}")
        lat_index, lon_index = get_rowcol_index_by_latlon(lat, lon)
        if logger:
            logger.info(f"lat_index, lon_index: {lat_index}, {lon_index}")
    else:
        lat_index = lon_index = None

    if lat_range is not None and lon_range is not None:
        if logger:
            logger.info(f"requested lat_range and lon_range: {lat_range}, {lon_range}")
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range
        min_row, max_row, min_col, max_col = get_rowcol_range_by_latlon(min_lat, max_lat, min_lon, max_lon)
        if logger:
            logger.info(f"min_row, max_row, min_col, max_col: {min_row}, {max_row}, {min_col}, {max_col}")
        lat_range = (min_row, max_row)
        lon_range = (min_col, max_col)
        if logger:
            logger.info(f"lat_range, lon_range: {lat_range}, {lon_range}")

    assert os.path.exists(path), f"File not found: {path}"
    assert os.access(path, os.R_OK), f"File not readable: {path}"
    if logger:
        logger.info(f"Reading data from {path} at address: {address}")
    try:
        with h5py.File(path, 'r') as f:
            if logger:
                logger.info(f"{path} opened successfully")
            if lat_index is not None and lon_index is not None:
                data = f[address][lat_index, lon_index]
                if "CDL" in address:
                    data = CDL_lookup(data)
                dict_data = {"value": data}
            elif lat_range is not None and lon_range is not None:
                data = f[address][lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
                data = np.where(data == -999, np.nan, data)  # Replace invalid values with NaN
                dict_data = process_data(data, address)
            else:
                data = f[address][:]
    except Exception as e:
        if logger:
            logger.error(f"Error reading data: {e}")
        return None
    if logger:
        logger.info(f"Data read successfully: {data}")    
    return dict_data

def process_data(data, address):
	data_median = np.nanmedian(data)
	data_max = np.nanmax(data)
	data_min = np.nanmin(data)
	data_mean = np.nanmean(data)
	data_std = np.nanstd(data)

	if "CDL" in address:
		unique, counts = np.unique(data, return_counts=True)
		cell_area_ha = 6.25
		dict_data = {
		    CDL_lookup(key): value * cell_area_ha
		    for key, value in zip(unique, counts)
		}
		dict_data |= {
			"Total Area": np.nansum(list(dict_data.values())),
			"unit": "hectares",
		}
	else:
		dict_data = {
		    "number of cells": data.size,
		    "median": data_median.round(2),
		    "max": data_max.round(2),
		    "min": data_min.round(2),
		    "mean": data_mean.round(2),
		    "std": data_std.round(2)
		}
	return dict_data

def get_rowcol_range_by_latlon(desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
    path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
    with h5py.File(path, 'r') as f:
        lat_ = f["geospatial/lat_250m"][:]
        lon_ = f["geospatial/lon_250m"][:]
        lat_ = np.where(lat_ == -999, np.nan, lat_)
        lon_ = np.where(lon_ == -999, np.nan, lon_)

        lat_mask = (lat_ >= desired_min_lat) & (lat_ <= desired_max_lat)
        lon_mask = (lon_ >= desired_min_lon) & (lon_ <= desired_max_lon)
        combined_mask = lat_mask & lon_mask

        if np.any(combined_mask):
            return _extract_rowcol_range(combined_mask)
        print("No valid points found for the given latitude and longitude range.")
        return None, None, None, None

def _extract_rowcol_range(combined_mask):
    row_indices, col_indices = np.where(combined_mask)
    min_row_number = np.min(row_indices)
    max_row_number = np.max(row_indices)
    min_col_number = np.min(col_indices)
    max_col_number = np.max(col_indices)
    return min_row_number, max_row_number, min_col_number, max_col_number

def get_rowcol_index_by_latlon(desired_lat, desired_lon):
    path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
    with h5py.File(path, 'r') as f:
        lat_ = f["geospatial/lat_250m"][:]
        lat_ = np.where(lat_ == -999, np.nan, lat_)
        lon_ = f["geospatial/lon_250m"][:]
        lon_ = np.where(lon_ == -999, np.nan, lon_)

        valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
        valid_lat = lat_[valid_mask]
        valid_lon = lon_[valid_mask]

        coordinates = np.column_stack((valid_lat, valid_lon))
        tree = cKDTree(coordinates)
        distance, idx = tree.query([desired_lat, desired_lon])

        valid_indices = np.where(valid_mask)
        lat_idx = valid_indices[0][idx]
        lon_idx = valid_indices[1][idx]

        return lat_idx, lon_idx

def single_model_creation(site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples):
    logging.info(f"Starting model creation for site_no: {site_no}")
    
    BASE_PATH = os.getenv('BASE_PATH', '/data/SWATGenXApp/GenXAppData/')
    from SWATGenX.SWATGenXLogging import LoggerSetup
    logger = LoggerSetup('/data/SWATGenXApp/codes/web_application/logs/', verbose=True, rewrite=False)
    logger.setup_logger(name="WebAppLogger")
    config = {
        "BASE_PATH": BASE_PATH,
        "LEVEL": "huc12",
        "MAX_AREA": 5000,
        "MIN_AREA": 10,
        "GAP_percent": 10,
        "landuse_product": "NLCD",
        "landuse_epoch": "2021",
        "ls_resolution": ls_resolution,
        "dem_resolution": dem_resolution,
        "station_name": site_no,
        "MODEL_NAME": 'SWAT_MODEL_Web_Application',
        "single_model": True,
        "sensitivity_flag": sensitivity_flag,
        "calibration_flag": calibration_flag,
        "verification_flag": validation_flag,
        "START_YEAR": 2015,
        "END_YEAR": 2022,
        "nyskip": 3,
        "sen_total_evaluations": sen_total_evaluations,
        "sen_pool_size": sen_pool_size,
        "num_levels": num_levels,
        "cal_pool_size": cal_pool_size,
        "max_cal_iterations": max_cal_iterations,
        "termination_tolerance": 10,
        "epsilon": 0.0001,
        "Ver_START_YEAR": 2004,
        "Ver_END_YEAR": 2022,
        "Ver_nyskip": 3,
        "range_reduction_flag": False,
        "pet": 2,
        "cn": 1,
        "no_value": 1e6,
        "verification_samples": verification_samples
    }

    logger.info(f"Configuration: {config}")
    # Model creation
    commander = SWATGenXCommand(config)
    model_path = commander.execute()

    # Calibration, validation, sensitivity analysis
    #if calibration_flag or validation_flag or sensitivity_flag:
    #    process_SCV_SWATGenXModel(config)

    # Output archive
    output_path = os.path.join("/data/Generated_models", f"{site_no}")
    os.makedirs("/data/Generated_models", exist_ok=True)
    try:
        shutil.make_archive(output_path, 'zip', model_path)
    except Exception as e:
        logging.error(f"Model creation failed for site_no: {site_no}")
    logger.info(f"Model creation successful for site_no: {site_no}")
    
    return f"{output_path}.zip"

def get_huc12_geometries(list_of_huc12s):
    #logging.info(f"Getting geometries for HUC12s: {list_of_huc12s}")
    ### list of huc12 are like:  ['020200030604', '020200030603', '020200030601', '020200030602', '020200030605']
    VPUID = list_of_huc12s[0][:4]
    print(VPUID)
    # Read the shapefile
    path = f"/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
    geodata_path = os.listdir(path)
    geodata_path = [x for x in geodata_path if x.endswith('.gdb')][0]
    geodata_path = os.path.join(path, geodata_path)
    gdf = gpd.read_file(geodata_path, layer = "WBDHU12")
    gdf.rename(columns = {'HUC12': 'huc12'}, inplace = True)
    gdf = gdf[gdf['huc12'].isin(list_of_huc12s)]
    return gdf['geometry'].apply(mapping).tolist()


def find_station(search_term='metal'):

    df = find_station_region(search_term)

    print(df)
    return df



if __name__ == '__main__':

    find_station(search_term='metal')