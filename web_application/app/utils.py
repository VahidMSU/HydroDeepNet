import sys
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
sys.path.append('/data/SWATGenXApp/codes/GeoReporter')
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
import traceback

HYDROGEO_DATASET_PATH = "/data/SWATGenXApp/GenXAppData/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
NHD_VPUID_PATH = "/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus"
CDL_LOOKUP_PATH = "/data/SWATGenXApp/GenXAppData/CDL/CDL_CODES.csv"
LOG_PATH = "/data/SWATGenXApp/codes/web_application/logs/"
BASE_PATH = "/data/SWATGenXApp/GenXAppData/"
USER_PATH = "/data/SWATGenXApp/Users/"


def check_existing_models(station_name,config):
	swatgenx_output = config.swatgenx_outlet_path
	VPUIDs = os.listdir(swatgenx_output)
	existing_models = []
	for VPUID in VPUIDs:
		# now find model inside huc12 directory
		huc12_path = os.path.join(swatgenx_output, VPUID, "huc12")
		models = os.listdir(huc12_path)
		existing_models.extend(os.path.join(huc12_path, model) for model in models)
	existance_flag = False
	for model in existing_models:
		if station_name in model:
			print(f"Model found for station {station_name} at {model}")
			existance_flag = True
			break
	return existance_flag

def send_verification_email(recipient):
    import smtplib
    from email.mime.text import MIMEText
    import uuid

    """
    Sends a verification email with a unique code to the specified recipient.

    Args:
        recipient (str): The email address of the recipient.

    Returns:
        str: The verification code sent in the email.
    """
    # Define the sender and recipient
    sender = "no-reply@ciwre.msu.edu"

    # Create the email content
    subject = "Verification Email"
    verification_code = str(uuid.uuid4().int)[:8]
    body = f"Your verification code is: {verification_code}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Define the SMTP server and port
    smtp_server = "express.mail.msu.edu"
    smtp_port = 25

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(sender, [recipient], msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

    return verification_code

import os
import logging

class LoggerSetup:
    def __init__(self, report_path: str, verbose: bool = True, rewrite: bool = False):
        """
        Initialize the LoggerSetup class.

        Args:
            report_path (str): Path to the directory where the log file will be saved.
            verbose (bool): Whether to print logs to console. Defaults to True.
            rewrite (bool): Whether to rewrite the log file if it already exists. Defaults to False.
        """
        self.report_path = report_path
        self.verbose = verbose
        self.rewrite = rewrite
        self.logger = None  # Placeholder for the logger instance

    def setup_logger(self, name: str = "GeoClassCNNLogger") -> logging.Logger:
        """
        Set up the logger to log messages to a file and optionally to the console.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if self.logger is None:
            log_file_path = os.path.join(self.report_path, f"{name}.log")

            # Delete existing log file if rewrite mode is enabled
            if self.rewrite and os.path.exists(log_file_path):
                os.remove(log_file_path)

            # Create the logger
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.INFO)

            # Define log format with enforced timestamp
            log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            # File handler for writing logs to a file
            file_handler = logging.FileHandler(log_file_path)
            self._extracted_from_setup_logger_31(file_handler, log_format)
            # Console handler for logging to the console
            if self.verbose:
                console_handler = logging.StreamHandler()
                self._extracted_from_setup_logger_31(console_handler, log_format)
            # Log the logger initialization path
            self.logger.info(f"Logger initialized: {log_file_path}")

        return self.logger

    # TODO Rename this here and in `setup_logger`
    def _extracted_from_setup_logger_31(self, arg0, log_format):
        arg0.setLevel(logging.INFO)
        arg0.setFormatter(log_format)
        self.logger.addHandler(arg0)

    def log(self, message: str, level: str = "info"):
        """
        Log a message with a specific logging level.

        Args:
            message (str): The message to log.
            level (str): The logging level (e.g., "info", "error", "warning", "debug").
        """
        if self.logger is None:
            raise RuntimeError("Logger is not initialized. Call `setup_logger()` first.")

        # Log message with timestamp (highest priority)
        log_methods = {
            "info": self.logger.info,
            "error": self.logger.error,
            "warning": self.logger.warning,
            "debug": self.logger.debug,
        }
        log_method = log_methods.get(level.lower(), self.logger.info)
        log_method(message)

    def error(self, message: str):
        """Log an error message."""
        self.log(message, level="error")

    def warning(self, message: str):
        """Log a warning message."""
        self.log(message, level="warning")

    def info(self, message: str):
        """Log an info message."""
        self.log(message, level="info")

    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, level="debug")


def hydrogeo_dataset_dict(path=HYDROGEO_DATASET_PATH):
    with h5py.File(path, 'r') as f:
        groups = f.keys()
        hydrogeo_dict = {group: list(f[group].keys()) for group in groups}
    return hydrogeo_dict

def CDL_lookup(code):
    path = CDL_LOOKUP_PATH
    df = pd.read_csv(path)
    df = df[df['CODE'] == code]
    return df.NAME.values[0]

def read_h5_file(address, lat=None, lon=None, lat_range=None, lon_range=None, logger=None):
    path = HYDROGEO_DATASET_PATH

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
    path = HYDROGEO_DATASET_PATH
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
    path = HYDROGEO_DATASET_PATH
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


def find_VPUID(station_no):
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths  
    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    return CONUS_streamflow_data[
        CONUS_streamflow_data.site_no == station_no
    ].huc_cd.values[0][:4]


def single_swatplus_model_creation(username, site_no, ls_resolution, dem_resolution):
    
    """ 
    Create a SWATGenX model for a single USGS site for a given user setting.
    """
    
    VPUID = find_VPUID(site_no)
    from SWATGenX.SWATGenXLogging import LoggerSetup
    logger = LoggerSetup(LOG_PATH, verbose=True, rewrite=False)
    logger.setup_logger(name="WebAppLogger")
    config = {
        "VPUID": VPUID,
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
        "sensitivity_flag": False,
        "calibration_flag": False,
        "verification_flag": False,
        "START_YEAR": 2015,
        "END_YEAR": 2022,
        "nyskip": 3,
        "sen_total_evaluations": 1000,
        "sen_pool_size": 10,
        "num_levels": 5,
        "cal_pool_size": 50,
        "max_cal_iterations": 50,
        "termination_tolerance": 0.0001,
        "epsilon": 0.0001,
        "Ver_START_YEAR": 2004,
        "Ver_END_YEAR": 2022,
        "Ver_nyskip": 3,
        "range_reduction_flag": False,
        "pet": 2,
        "cn": 1,
        "no_value": 1e6,
        "verification_samples": 5,
        "username": username,
    }

    logger.info(f"Configuration: {config}")
    # Model creation
    
    os.makedirs(f"{USER_PATH}/{username}/SWATplus_by_VPUID", exist_ok=True)
    
    if not os.path.exists(f"{USER_PATH}/{username}/SWATplus_by_VPUID/"):
         logger.error(f"Output directory not found: {USER_PATH}/{username}/SWATplus_by_VPUID/")
    else:
        logger.info(f"Output directory found: {USER_PATH}/{username}/SWATplus_by_VPUID/")
    
    # Initialize model_path to None in case of exceptions
    model_path = None
    
    try:
        commander = SWATGenXCommand(config)
        model_path = commander.execute()
        logger.info(f"CommandX: Model created successfully: {model_path}") 
    except Exception as e:
        logger.error(f"Error in single_swatplus_model_creation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Set a default path or failure indicator
        expected_path = f"{USER_PATH}/{username}/SWATplus_by_VPUID/{VPUID}/huc12/{site_no}/SWAT_MODEL_Web_Application"
        logger.error(f"Model would have been created at: {expected_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
       
    # Calibration, validation, sensitivity analysis
    #if calibration_flag or validation_flag or sensitivity_flag:
    #    process_SCV_SWATGenXModel(config)

    # Output archive
    return model_path

def get_huc12_geometries(list_of_huc12s):

    VPUID = list_of_huc12s[0][:4]
    print(VPUID)
    # Read the shapefile
    path = f"{NHD_VPUID_PATH}/{VPUID}/unzipped_NHDPlusVPU/"
    geodata_path = os.listdir(path)
    geodata_path = [x for x in geodata_path if x.endswith('.gdb')][0]
    geodata_path = os.path.join(path, geodata_path)
    gdf = gpd.read_file(geodata_path, layer = "WBDHU12")
    gdf.rename(columns = {'HUC12': 'huc12'}, inplace = True)
    gdf = gdf[gdf['huc12'].isin(list_of_huc12s)]
    return gdf['geometry'].apply(mapping).tolist()

def get_huc12_streams_geometries(list_of_huc12s):

    VPUID = list_of_huc12s[0][:4]
    print(VPUID)
    # Read the shapefile
    path = f"{NHD_VPUID_PATH}/{VPUID}/streams.pkl"
    
    gdf = gpd.GeoDataFrame(pd.read_pickle(path)).to_crs("EPSG:4326")
    ### make sure list_of_huc12s and huc12 column in the geodataframe are in 12 digit int
    gdf['huc12'] = gdf['huc12'].astype(int)
    list_of_huc12s = [int(x) for x in list_of_huc12s]
    
    ### type as str
    gdf = gdf[gdf['huc12'].isin(list_of_huc12s)]
    WBArea_Permanent_Identifier = gdf['WBArea_Permanent_Identifier'].tolist() 
    
    return gdf['geometry'].apply(mapping).tolist(), WBArea_Permanent_Identifier

def get_huc12_lakes_geometries(list_of_huc12s, WBArea_Permanent_Identifier):
    VPUID = list_of_huc12s[0][:4]
    path = f"{NHD_VPUID_PATH}/{VPUID}/NHDWaterbody.pkl"
    gdf = gpd.GeoDataFrame(pd.read_pickle(path)).to_crs("EPSG:4326")

    # Rename column to match your usage
    gdf = gdf.rename(columns={'Permanent_Identifier': 'WBArea_Permanent_Identifier'})
    gdf['WBArea_Permanent_Identifier'] = gdf['WBArea_Permanent_Identifier'].astype(str)

    gdf = gdf[gdf.WBArea_Permanent_Identifier.isin(WBArea_Permanent_Identifier)] 
    return gdf['geometry'].apply(mapping).tolist()

def find_station(search_term='metal'):
    df = find_station_region(search_term)
    print(df)
    return df

if __name__ == '__main__':

    find_station(search_term='metal')