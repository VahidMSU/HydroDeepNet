from calendar import c
from datetime import datetime
from logging import config
import h5py
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
import matplotlib.animation as animation
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def hydrogeo_dataset_dict(path=None):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path,'r') as f:
		groups = f.keys()
		hydrogeo_dict = {}
		for group in groups:	
			hydrogeo_dict[group] = list(f[group].keys())
	return hydrogeo_dict


def read_h5_file(address, lat=None, lon=None, lat_range=None, lon_range=None):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path, 'r') as f:
		if lat is not None and lon is not None:
			data = f[address][lat, lon]
		elif lat_range is not None and lon_range is not None:
			data = f[address][lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]
		else:
			data = f[address][:]
	return data


def get_rowcol_range_by_latlon(desired_min_lat, desired_max_lat, desired_min_lon, desired_max_lon):
	path = "/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
	with h5py.File(path, 'r') as f:
		# Read latitude and longitude arrays
		lat_ = f["geospatial/lat_250m"][:]
		lon_ = f["geospatial/lon_250m"][:]
		
		# Replace missing values (-999) with NaN for better handling
		lat_ = np.where(lat_ == -999, np.nan, lat_)
		lon_ = np.where(lon_ == -999, np.nan, lon_)

		# Create masks for latitude and longitude ranges
		lat_mask = (lat_ >= desired_min_lat) & (lat_ <= desired_max_lat)
		lon_mask = (lon_ >= desired_min_lon) & (lon_ <= desired_max_lon)

		# Combine the masks to identify the valid rows and columns
		combined_mask = lat_mask & lon_mask

		# Check if any valid points are found
		if np.any(combined_mask):
			# Get row and column indices where the combined mask is True
			row_indices, col_indices = np.where(combined_mask)
		else:
			print("No valid points found for the given latitude and longitude range.")

		min_row_number = np.min(row_indices)
		max_row_number = np.max(row_indices)
		min_col_number = np.min(col_indices)
		max_col_number = np.max(col_indices)
		

		print(f"Min row number: {min_row_number}, Max row number: {max_row_number}, Min column number: {min_col_number}, Max column number: {max_col_number}")

		return min_row_number, max_row_number, min_col_number, max_col_number





def get_rowcol_index_by_latlon(desired_lat, desired_lon, RESOLUTION=250):
	path = f"/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_{RESOLUTION}.h5"

	with h5py.File(path, 'r') as f:
		lat_ = f["geospatial/lat_250m"][:]
		lat_ = np.where(lat_ == -999, np.nan, lat_)  # Replace invalid values with NaN
		lon_ = f["geospatial/lon_250m"][:]
		lon_ = np.where(lon_ == -999, np.nan, lon_)  # Replace invalid values with NaN

		valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
		valid_lat = lat_[valid_mask]
		valid_lon = lon_[valid_mask]

		# Stack valid coordinates into KDTree
		coordinates = np.column_stack((valid_lat, valid_lon))
		tree = cKDTree(coordinates)

		# Query the closest point
		distance, idx = tree.query([desired_lat, desired_lon])

		# Retrieve the original indices of the closest point
		valid_indices = np.where(valid_mask)
		lat_idx = valid_indices[0][idx]
		lon_idx = valid_indices[1][idx]

		print(f"Closest row: {lat_idx}, Closest column: {lon_idx}")

		# Check the latitude and longitude values for the generated row and column
		lat_val = lat_[lat_idx, lon_idx]
		lon_val = lon_[lat_idx, lon_idx]

		print(f"Latitude: {lat_val}, Longitude: {lon_val}")

		return lat_idx, lon_idx



def list_of_cc_models(required_models="MPI-ESM1-2-HR", required_scenarios="historical"):

	"""
	
	Read and return the list of climate models, scenarios, and ensembles in LOCA2 
	
	"""
	# Climate model, scenario, and ensemble configuration for LOCA2
	list_of_climate_data = '/data/LOCA2/list_of_all_models.txt'

	dict_of_cc_models = {'cc_model': [], 'scenario': [], 'ensemble': []}
	with open(list_of_climate_data, 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			parts = line.split(' ')
			if len(parts) == 4:
				idx, cc_model, scenario, ensemble = parts
				dict_of_cc_models['cc_model'].append(cc_model)
				dict_of_cc_models['scenario'].append(scenario)
				dict_of_cc_models['ensemble'].append(ensemble)
				#print(cc_model, scenario, ensemble)
			#else:

				#print(f"Skipping line: {line}")
		## create a dataframe

		df = pd.DataFrame(dict_of_cc_models)[:99]
		
		df2 = df[:99][df.scenario == 'historical']


	return df2

class DataImporter:
	def __init__(self, config, device=None):
		"""
		Summary:
		Class for importing and processing various types of data for hydrogeological modeling.

		Explanation:
		This class provides methods for importing and processing different types of data including static, transient, and PFAS data. It handles tasks such as extracting features and preparing data for deep learning.

		"""
		
		self.config = config if isinstance(config, dict) else config.__dict__
		self.device = device
		self.config['RESOLUTION'] = 250 if 'RESOLUTION' not in config else config['RESOLUTION']
		self.config['database_path'] = f'/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_{config["RESOLUTION"]}.h5'
		self.config['geoloc'] = False if 'geoloc' not in config else config['geoloc']
		self.config['snow'] = False if 'snow' not in config else config['snow']
		self.config['groundwater'] = False if 'groundwater' not in config else config['groundwater']
		self.config['population_array'] = False if 'population_array' not in config else config['population_array']
		self.config['landfire'] = False if 'landfire' not in config else config['landfire']
		self.config['geology'] = False if 'geology' not in config else config['geology']
		self.config['NHDPlus'] = False if 'NHDPlus' not in config else config['NHDPlus']
		self.config['plot'] = False if 'plot' not in config else config['plot']
		self.config['pfas_database_path'] = f'/data/MyDataBase/HydroGeoDataset/PFAS_sw_{config["RESOLUTION"]}m.h5'
		self.config['huc8'] = None if 'huc8' not in config else config['huc8']
		self.config['snowdas_h5_path'] = '/data/MyDataBase/HydroGeoDataset/SNODAS.h5'
		self.config['video'] = False if 'video' not in config else config['video']
		self.config['aggregation'] = None if 'aggregation' not in config else config['aggregation']

	def get_database_rows_cols(self):

		""" Get the rows and columns of the database. """
		path = f"/data/MyDataBase/HydroGeoDataset/HydroGeoDataset_ML_{self.config['RESOLUTION']}.h5"
		with h5py.File(path, 'r') as f:
			reference = f['geospatial/DEM_250m'][:]
			rows = reference.shape[0]
			cols = reference.shape[1]
			print(f"rows: {rows}, cols: {cols}")
			

		return rows, cols
	
	def get_var_name(self, feature_type, RESOLUTION, config):
		features = []
		if feature_type == 'categorical':

			features = [
				f'geospatial/COUNTY_{self.config["RESOLUTION"]}m',
				f'geospatial/landforms_{config["RESOLUTION"]}m_250Dis',
				f'Soil/gSURRGO_swat_{self.config["RESOLUTION"]}m',
				f'geospatial/landuse_{self.config["RESOLUTION"]}m',
				f'geospatial/geomorphons_{config["RESOLUTION"]}m_250Dis',
				f'geospatial/MI_geol_poly_{self.config["RESOLUTION"]}m',
				f'geospatial/Glacial_Landsystems_{self.config["RESOLUTION"]}m',
				f'geospatial/Aquifer_Characteristics_Of_Glacial_Drift_{self.config["RESOLUTION"]}m',
				]
		elif feature_type == 'numerical':
			self.select_numerical_features(config, features, RESOLUTION)
		return features

	def select_numerical_features(self, config, features, RESOLUTION) -> None:
		if config.get("geospatial", False):
			features.extend([
				f'geospatial/DEM_{self.config["RESOLUTION"]}m',
				f'geospatial/x_{self.config["RESOLUTION"]}m',
				f'geospatial/y_{self.config["RESOLUTION"]}m',
			])

		if config.get("climate_pattern", False):

			features.extend([
					f'climate_pattern/non_snow_accumulation_raster_{self.config["RESOLUTION"]}m',
					f'climate_pattern/snow_accumulation_raster_{self.config["RESOLUTION"]}m',
					f'climate_pattern/melt_rate_raster_{self.config["RESOLUTION"]}m',
					f'climate_pattern/average_temperature_raster_{self.config["RESOLUTION"]}m',
					f'climate_pattern/snow_layer_thickness_raster_{self.config["RESOLUTION"]}m',
				])

		if config.get("EBK", False):
			features.extend([
				f'EBK/kriging_output_H_COND_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_AQ_THK_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_H_COND_2_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_SWL_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_V_COND_2_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_TRANSMSV_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_TRANSMSV_2_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_output_V_COND_1_{self.config["RNHDPlus/ESOLUTION"]}m',
				f'EBK/kriging_stderr_SWL_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_H_COND_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_H_COND_2_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_V_COND_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_V_COND_2_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_AQ_THK_1_{self.config["RESOLUTION"]}m',
				f'EBK/kriging_stderr_AQ_THK_2_{self.config["RESOLUTION"]}m',
			])

		if config.get("NHDPlus", False):
			features.extend([
				f'NHDPlus/QAMA_MILP_{self.config["RESOLUTION"]}m',        ## mean annual streamflow
				f'NHDPlus/QBMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow from excess ET
				f'NHDPlus/QCMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow with reference gage regression
				f'NHDPlus/QDMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow with NHDPlusAdditionRemoval
				f'NHDPlus/QEMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow from gage adjustment
				f'NHDPlus/QIncrBMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow with excess ET
				f'NHDPlus/QIncrCMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow by subtracting upstream QCMA
				f'NHDPlus/QFMA_MILP_{self.config["RESOLUTION"]}m',
				f'NHDPlus/QGAdjMA_MILP_{self.config["RESOLUTION"]}m',
				f'NHDPlus/QIncrAMA_MILP_{self.config["RESOLUTION"]}m',
				f'NHDPlus/QIncrDMA_MILP_{self.config["RESOLUTION"]}m', 	  ## Incremental flow with NHDPlusAdditionRemoval
				f'NHDPlus/QIncrEMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow from gage adjustment
				f'NHDPlus/QIncrFMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow from gage sequestration
				f'NHDPlus/VBMA_MILP_{self.config["RESOLUTION"]}m',        # Velocity for QBMA
				f'NHDPlus/VCMA_MILP_{self.config["RESOLUTION"]}m',		  # Velocity for QCMA
				f'NHDPlus/VDMA_MILP_{self.config["RESOLUTION"]}m',        # Velocity for QCMA
				f'NHDPlus/VEMA_MILP_{self.config["RESOLUTION"]}m',	      # Velocity from gage adjustment
			])
		if config.get('LANDFIRE', False):	
			features.extend([
				f'LADFIRE/LC20_Asp_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC20_BPS_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC20_EVT_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC20_Elev_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC20_SlpD_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC20_SlpP_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC22_EVC_220_{self.config["RESOLUTION"]}m',
				f'LADFIRE/LC22_EVH_220_{self.config["RESOLUTION"]}m',
			])

		if config.get('population', False):
			features.extend([
				f'population/pden1990_ML_{self.config["RESOLUTION"]}m',
				f'population/pden2000_ML_{self.config["RESOLUTION"]}m',
				f'population/pden2010_ML_{self.config["RESOLUTION"]}m',
			])

	def gw_3d_ds(self, start_year=2020, end_year=2022) -> np.ndarray:
		""" Extract the 3D groundwater head data from the database. """

		if self.config['RESOLUTION'] != 250:
			raise ValueError("Groundwater head data is only available at 250m RESOLUTION.")
		path = '/data/MyDataBase/HydroGeoDataset/gw_head.h5'
		with h5py.File(path, 'r') as f:
			## NOTE: date from 1-1-1990 to 12-31-2022
			start_index = (start_year - 1990) * 365
			end_index = (end_year - 1990) * 365

			if self.config['huc8']:
				row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				gw_head = f['gw_head'][start_index:end_index, row_min:row_max, col_min:col_max]
			else:
				gw_head = f['gw_head'][start_index:end_index]

			gw_head = np.where(gw_head == -999, np.nan, gw_head)
			## if all nan, return None
			if np.isnan(gw_head).all():
				logging.info("All nan values in the groundwater head data.")
				return None

		logging.info(f"Groundwater head shape: {gw_head.shape}")
		return gw_head

	## add hints for the stations
	def gw_stations_ds(self, stations=None, start_year=2020, end_year=2022) -> np.ndarray:
		""" Extract the groundwater head data for specific stations from the database. """

		path = '/data/MyDataBase/HydroGeoDataset/gw_head_2d.h5'

		numerical, categorical, _ = self.import_static_data(huc8=False)
		print(f"Numerical shape: {numerical.shape}, Categorical shape: {categorical.shape}")
		with h5py.File(path, 'r') as f:
			## NOTE: date from 1-1-1990 to 12-31-2022
			start_index = (start_year - 1990) * 365
			end_index = (end_year - 1990) * 365
			gw_station_data = {}
			#if stations is None:
			stations = f.keys()
			#logging.info(f"## stations: {stations}")
			for station in stations:

				gw_head = f[station][start_index:end_index]
				#logging.info(f"## station: {station} with shape: {gw_head.shape} and %{100*(1- sum(np.isnan(gw_head)/gw_head.size)):.2f} observations. ")
				row = station.split('_')[1]
				col = station.split('_')[2]
				numerical_feature =  numerical[:, int(row), int(col)]
				categorical_feature = categorical[:, int(row), int(col)]

				## if all nan, return None
				if np.isnan(gw_head).all():
					logging.info(f"All nan values in the groundwater head data for station {station}.")
					continue
				### add the features to the dictionary
				gw_station_data[station] = {
					'gw_head': gw_head,
					'numerical_feature': numerical_feature,
					'categorical_feature': categorical_feature,
				}
		#logging.info(f"Groundwater head data for stations: {list(gw_station_data.keys())}")
		#logging.info(f"Groundwater head data for stations: {list(gw_station_data.keys())}")
		#logging.info(f"Groundwater head data for stations: {list(gw_station_data.keys())}")

		return gw_station_data

	def tensorize(self, array3d):
		percentile_99 = np.percentile(array3d[array3d != -999], 99)
		array3d = np.where(array3d > percentile_99, -999, array3d)

		return array3d



	def MODIS_ET(self, start_year=None, end_year=None, h5_group_name="MODIS_ET"):
		"""
		Extract MODIS ET data for a given period (start_year to end_year).
		
		:param start_year: Starting year of the period
		:param end_year: Ending year of the period
		:param h5_group_name: The group name inside the HDF5 file that contains the MODIS ET data (default is "MODIS_ET")
		:return: Numpy array containing the extracted data for the specified period
		"""

		extracted_data = []
		if start_year is None:
			assert 'start_year' in self.config, "start_year is not provided."
			start_year = self.config['start_year']
		if end_year is None:
			assert 'end_year' in self.config, "end_year is not provided."
			end_year = self.config['end_year']


		## get huc8 ranges
		if self.config['huc8']:
			min_x, max_x, min_y, max_y = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
			logging.info(f"HydroGeoDataSet Range of Lat and Lon for the Required huc8: {min_x, max_x, min_y, max_y}")


		# Open the HDF5 file
		print(f"Opening the HDF5 file: {self.config['database_path']}")	
		with h5py.File(self.config['database_path'], "r") as f:
		
			# Get the list of datasets
			datasets = list(f[h5_group_name].keys())
			print(f"Total number of datasets: {len(datasets)}")

			# Loop over the specified years and extract data for each dataset in the period
			for year in range(start_year, end_year + 1):
				for month in range(1, 13):
					dataset_name_patter = f"MODIS_ET_{year}-{month:02d}"
					for dataset_name in datasets:
						if dataset_name.startswith(dataset_name_patter):
							print(f"Extracting dataset: {dataset_name}")
							if self.config['huc8']:
								img = f[f"{h5_group_name}/{dataset_name}"][min_x:max_x, min_y:max_y]	
							else:

								img = f[f"{h5_group_name}/{dataset_name}"][:]
							extracted_data.append(img)

		# Stack the list of images into a single NumPy array
		if len(extracted_data) > 0:
			extracted_data = np.stack(extracted_data, axis=0)  # Stack along a new axis (time or depth)
		else:
			extracted_data = np.array([])  # Return an empty array if no data is found

		print(f"Total number of datasets extracted: {len(extracted_data)}")
		print(f"Shape of the extracted data: {extracted_data.shape}")
		
		return extracted_data



	def extract_snowdas_data(self, snowdas_var, year) -> np.ndarray:

		""" parameter to extract the data from the SNODAS dataset.
		Parameters:
			'average_temperature', 'melt_rate', 'non_snow_accumulation', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate'
		Geographic extent:
			Michigan LP

		Example:
		extract_snowdas_data('snowdas_var', 44, 47, -87, -84)
		"""
		with h5py.File(self.config["snowdas_h5_path"], "r") as h5_file:
			logging.info(f"Extracting {snowdas_var} data for the year {year}.")

			var = h5_file[f"250m/{year}/{snowdas_var}"][:]
			unit = h5_file[f"250m/{year}/{snowdas_var}"].attrs['units']
			logging.info(f"Size of the SNOWDAS data: {var.shape}")
			convertor = h5_file[f"250m/{year}/{snowdas_var}"].attrs['converters']
			var = np.where(var == 55537, np.nan, var*convertor)
			if self.config['huc8']:

				min_x, max_x, min_y, max_y  = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				var = var[:, min_x:max_x, min_y:max_y]


			logging.info(f"Size of the SNOWDAS data after cropping: {var.shape}")
			if self.config['video']:
				self.video_data(var, f"{snowdas_var}_{unit}_{year}")
		return var

	def video_data(self, data, name, number_of_frames=None) -> None:
		"""
		Create a video from a 3D data array where each 2D array represents a frame.

		Parameters:
		data (numpy.ndarray): 3D array with the shape (frames, height, width).
		name (str): Name of the output video file.

		Returns:
		None
		"""
		logging.info(f"Creating video of {name} data.")
		# Replace -999 with NaN for better visualization
		data = np.where(data == -999, np.nan, data)

		# Set up the figure and axis
		fig, ax = plt.subplots()

		# Calculate global min and max values for the entire time series (all frames)
		vmin = np.nanpercentile(data, 2.5)  # 2.5th percentile value across all frames
		vmax = np.nanpercentile(data, 97.5)  # 97.5th percentile value across all frames

		# Create the first image with the full time series color limits (vmin, vmax)
		im = ax.imshow(data[0], animated=True, cmap='viridis', vmin=vmin, vmax=vmax)

		# Add the color bar (created only once, with limits based on full time series)
		cbar = fig.colorbar(im, ax=ax)
		## data unit
		if 'ppt' in name or 'pr' in name:
			dataunit = 'mm/day'
		else:
			dataunit = 'C'
		cbar.set_label(f'95th Percentile Range ({dataunit})')

		def update_frame(i):
			"""Update the frame for the animation."""
			# Update the data of the image for the new frame without recreating colorbar
			im.set_array(data[i])
			title = name.split('_')[1] + " " + name.split("_")[0].upper()
			ax.set_title(f"{title}\nStep {i+1}: Mean: {np.nanmean(data[i]):.2f} 97.25th: {np.nanpercentile(data[i], 97.25):.2f} 2.5th: {np.nanpercentile(data[i], 2.5):.2f}")
			return [im]

		# Create animation
		ani = animation.FuncAnimation(fig, update_frame, frames=range(data.shape[0]), interval=50, blit=True, repeat_delay=1000)

		# Ensure the output directory exists
		os.makedirs('input_videos', exist_ok=True)

		# Determine the output file name
		
		output_filename = f'input_videos/{name}.gif'
		if self.config.get('huc8'):
			output_filename = f'input_videos/{name}_{self.config["huc8"]}.gif'

		# Save the animation
		ani.save(output_filename, writer='pillow')

		# Close the figure
		plt.close(fig)

		logging.info(f"Video of {name} data saved as {output_filename}.")



	@staticmethod
	def get_loca2_time_index_of_year(start_year, end_year):
		"""Get the start and end indices of the given years in the LOCA2 dataset."""
		LOCA2_start_date = datetime(1950, 1, 1)
		LOCA2_end_date = datetime(2014, 12, 31)

		# Calculate the start index for the given start_year
		extract_year_start = datetime(start_year, 1, 1)
		index_year_start = (extract_year_start - LOCA2_start_date).days + 1

		# Calculate the end index for the given end_year
		extract_year_end = datetime(end_year, 12, 31)
		index_year_end = (extract_year_end - LOCA2_start_date).days + 1

		# Print the results
		#print(f"Start index of {start_year}: {index_year_start}")
		#print(f"End index of {end_year}: {index_year_end}")

		# Return the indices
		return index_year_start, index_year_end+1



	def clip_h5(self, f, data):
		""" Clip the data to the extent of a given HUC8. """
		loca2_lats = f['lat'][:]
		loca2_lons = f['lon'][:]
		huc8_lat_max, huc8_lat_min, huc8_lon_max, huc8_lon_min = self.get_huc8_latlon(self.config['database_path'], self.config['huc8'])
		#logging.info(f"HydroGeoDataSet Range of Lat and Lon for the Required huc8: {huc8_lat_max, huc8_lat_min, huc8_lon_max, huc8_lon_min}")
		clipped_loca2_rows = np.where((loca2_lats >= huc8_lat_min) & (loca2_lats <= huc8_lat_max))[0]
		clipped_loca2_cols = np.where((loca2_lons >= huc8_lon_min) & (loca2_lons <= huc8_lon_max))[0]
		
		loca2_max_rows = np.max(clipped_loca2_rows)
		loca2_min_rows = np.min(clipped_loca2_rows)
		loca2_max_cols = np.max(clipped_loca2_cols)
		loca2_min_cols = np.min(clipped_loca2_cols)

		## get the loca2_max_rows, loca2_min_rows, loca2_max_cols, loca2_min_cols
		loca2_lats = loca2_lats[loca2_min_rows:loca2_max_rows]
		loca2_lons = loca2_lons[loca2_min_cols:loca2_max_cols]

		#logging.info(f"LOCA2 Clipped Range of Lat and Lon for the Required huc8: {np.nanmax(loca2_lats), np.nanmin(loca2_lats), np.nanmax(loca2_lons), np.nanmin(loca2_lons)}")



		clipped = data[:, loca2_min_rows:loca2_max_rows, loca2_min_cols:loca2_max_cols]

		print(f"LOCA2 clipped shape: {clipped.shape}")	

		return clipped
	

	def get_huc8_ranges(self, database_path, huc8_select) -> tuple:
		"""
		Summary:
		Get the row and column ranges of a specific HUC8 value from the HydroGeoDataset.

		Args:
			self: The instance of the class.
			database_path (str): The path to the h5 database file containing the HUC8 data.
			huc8_select (str): The specific HUC8 value to search for.

		Returns:
			tuple: A tuple containing the minimum and maximum row indices (row_min, row_max) and column indices (col_min, col_max) of the specified HUC8 value.
		"""
		print(f'attempting to open database: {database_path}')
		with h5py.File(database_path, 'r') as f:

			huc8 = np.array(f[f'NHDPlus/HUC8_{self.config["RESOLUTION"]}m'][:])

		rows, cols = np.where(huc8 == int(huc8_select))
		row_min, row_max = rows[rows!=-999].min(), rows.max()
		col_min, col_max = cols[cols!=-999].min(), cols.max()

		print(f"loaded row_min: {row_min}, row_max: {row_max}, col_min: {col_min}, col_max: {col_max}")	

		huc8_lat_max, huc8_lat_min, huc8_lon_max, huc8_lon_min = self.get_huc8_latlon(database_path, huc8_select)

		logging.info(f"HydroGeoDataSet{self.config['RESOLUTION']}: range of lat and lon for the required huc8: {huc8_lat_max, huc8_lat_min, huc8_lon_max, huc8_lon_min}")
		return row_min, row_max, col_min, col_max


	def get_huc8_latlon(self, database_path, huc8_select) -> tuple:
		"""
		Summary:
		Get the row and column ranges of a specific HUC8 value from the HydroGeoDataset.

		Args:
			self: The instance of the class.
			database_path (str): The path to the h5 database file containing the HUC8 data.
			huc8_select (str): The specific HUC8 value to search for.

		Returns:
			tuple: A tuple containing the minimum and maximum row indices (row_min, row_max) and column indices (col_min, col_max) of the specified HUC8 value.
		"""

		with h5py.File(database_path, 'r') as f:
			huc8 = np.array(f[f'NHDPlus/HUC8_{self.config["RESOLUTION"]}m'][:])
			huc8_lats = np.array(f[f'geospatial/lat_{self.config["RESOLUTION"]}m'][:])
			huc8_lons = np.array(f[f'geospatial/lon_{self.config["RESOLUTION"]}m'][:])

		#(np.float32(42.382362), np.float32(41.680576), np.float32(-84.445755), np.float32(-86.54118)
		#(np.float64(42.28125), np.float64(41.71875), np.float64(-84.53125), np.float64(-86.53125))

		rows, cols = np.where(huc8 == int(huc8_select))
		row_min, row_max = rows[rows!=-999].min(), rows.max()
		col_min, col_max = cols[rows!=-999].min(), cols.max()

		huc8_lat = huc8_lats[row_min:row_max, col_min:col_max]
		huc8_lon = huc8_lons[row_min:row_max, col_min:col_max]

		huc8_lat_max = np.nanmax(huc8_lat[huc8_lat!=-999])
		huc8_lat_min = np.nanmin(huc8_lat[huc8_lat!=-999])
		huc8_lon_max = np.nanmax(huc8_lon[huc8_lon!=-999])
		huc8_lon_min = np.nanmin(huc8_lon[huc8_lon!=-999])

		return huc8_lat_max, huc8_lat_min, huc8_lon_max, huc8_lon_min

	def LOCA2(self, start_year, end_year, cc_model, scenario, ensemble,cc_time_step, row = None, col = None) -> np.ndarray:
		self.config['start_year'] = start_year
		self.config['end_year'] = end_year
		if scenario == 'historical':
			time_range = '1950_2014'
		else:
			time_range = '2015_2100'
		""" The function imports the climate data from the database
		and applies the necessary preprocessing steps."""

		
		start, end = self.get_loca2_time_index_of_year(start_year, end_year)

		path = '/data/SWATGenXApp/LOCA2_MLP.h5'
		with h5py.File(path, 'r') as f:
			#logging.info(f"LOCA2 keys: {f.keys()}")
			## time length 
			mask = self.get_mask()
			#logging.info(f"LOCA2 time length: {f['e_n_cent/ACCESS-CM2/historical/r1i1p1f1/daily/1950_2014/pr'].shape[0]}")
			logging.info(f"Attempting to load climate data for {cc_model}, {scenario}, {ensemble}, {cc_time_step}, {time_range}.")
			pr = f[f'e_n_cent/{cc_model}/{scenario}/{ensemble}/{cc_time_step}/{time_range}/pr'][start:end]      # 3D array, shape: (23741, 67, 75)
			tmax = f[f'e_n_cent/{cc_model}/{scenario}/{ensemble}/{cc_time_step}/{time_range}/tasmax'][start:end] # 3D array, shape: (23741, 67, 75)
			tmin = f[f'e_n_cent/{cc_model}/{scenario}/{ensemble}/{cc_time_step}/{time_range}/tasmin'][start:end]
			if self.config['huc8']:
				
				pr = self.clip_h5(f, pr)
				tmax = self.clip_h5(f, tmax)
				tmin = self.clip_h5(f, tmin)

				

				rows = mask.shape[0]
				cols = mask.shape[1]
				#print(f"loaded rows: {rows}, cols: {cols}")

			else:

				logging.info(f"Size of the climate data: {pr.shape}, {tmax.shape}, {tmin.shape}")
				rows, cols = self.get_database_rows_cols()

			# Calculate the replication factors
			rep_factors = (int(np.ceil((rows-1) / pr.shape[1])), int(np.ceil((cols-1) / pr.shape[2])))
			logging.info(f"Replication factors: {rep_factors}")

			# Replicate the climate data using numpy.repeat
			pr = np.repeat(pr, rep_factors[0], axis=1)
			pr = np.repeat(pr, rep_factors[1], axis=2)

			tmax = np.repeat(tmax, rep_factors[0], axis=1)
			tmax = np.repeat(tmax, rep_factors[1], axis=2)

			tmin = np.repeat(tmin, rep_factors[0], axis=1)
			tmin = np.repeat(tmin, rep_factors[1], axis=2)

			logging.info(f"Size of the climate data after replication: {pr.shape}, {tmax.shape}, {tmin.shape}")

			# Flip the climate data to correct the orientation
			pr = np.flip(pr, axis=1).copy()
			tmax = np.flip(tmax, axis=1).copy()
			tmin = np.flip(tmin, axis=1).copy()
			logging.info("Flipping completed.")

			# Pad the climate data to achieve the exact target shape
			## shape before cropping: (23741, 67, 75)
			logging.info(f"Shape before cropping: {pr.shape}, {tmax.shape}, {tmin.shape}")
			

			target_shape = (pr.shape[0], rows, cols)
			pr = pr[:, :target_shape[1], :target_shape[2]]
			tmax = tmax[:, :target_shape[1], :target_shape[2]]
			tmin = tmin[:, :target_shape[1], :target_shape[2]]

			logging.info(f"Size of the climate data after cropping: {pr.shape}, {tmax.shape}, {tmin.shape}")
			## replace nan with -999

			### get the mask
			
			mask = mask[:pr.shape[1], :pr.shape[2]]  # Adjust mask shape to match pr shape
			## also convert kg m-2 s-1 to mm/day
			pr = np.where(mask != 1, -999, pr*86400)
			## also convet K to C
			tmax = np.where(mask != 1, -999, tmax - 273.15)
			tmin = np.where(mask != 1, -999, tmin - 273.15)
			


			### if there is any nan, convert it to -999
			pr = np.where(np.isnan(pr), -999, pr)
			tmax = np.where(np.isnan(tmax), -999, tmax)
			tmin = np.where(np.isnan(tmin), -999, tmin)

			if self.config['video']:
				self.video_data(pr, 'pr_LOCA2')
				self.video_data(tmax, 'tmax_LOCA2')
				self.video_data(tmin, 'tmin_LOCA2')


		
			if self.config['aggregation']:
				pr, tmax, tmin = self.aggregate_temporal_data(pr, tmax, tmin)
				logging.info(f"Aggregated data shape: {pr.shape}, {tmax.shape}, {tmin.shape}")

			return pr, tmax, tmin


	def aggregate_temporal_data(self, pr, tmax, tmin) -> np.ndarray:
		""" Aggregate the temporal data based on the specified aggregation method. """
		min_temporal = min(pr.shape[0], tmax.shape[0], tmin.shape[0])
		pr = pr[:min_temporal, :, :]
		tmax = tmax[:min_temporal, :, :]
		tmin = tmin[:min_temporal, :, :]
		print(f"#################Aggregation method: {self.config['aggregation']}#################")
		
		total_days = pr.shape[0]
		# Create a date range assuming data starts on January 1st of the start year
		start_date = f"{self.config['start_year']}-01-01"
		dates = pd.date_range(start=start_date, periods=total_days)
		
		if self.config['aggregation'] == 'monthly':
			# Group by both year and month using the dates array
			pr_monthly = [pr[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
						for year in np.unique(dates.year)
						for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
			
			tmax_monthly = [tmax[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
							for year in np.unique(dates.year)
							for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
			
			tmin_monthly = [tmin[(dates.year == year) & (dates.month == month), :, :].mean(axis=0)
							for year in np.unique(dates.year)
							for month in range(1, 13) if np.sum((dates.year == year) & (dates.month == month)) > 0]
			
			pr = np.array(pr_monthly)
			tmax = np.array(tmax_monthly)
			tmin = np.array(tmin_monthly)
			logging.info(f"Aggregated data shape (monthly): {pr.shape}, {tmax.shape}, {tmin.shape}")

		elif self.config['aggregation'] == 'seasonal':
			# Define the seasons as months: DJF, MAM, JJA, SON
			seasons = {
				'DJF': [12, 1, 2],
				'MAM': [3, 4, 5],
				'JJA': [6, 7, 8],
				'SON': [9, 10, 11]
			}

			pr_seasonal = [pr[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
						for year in np.unique(dates.year)
						for season, months in seasons.items()]
			
			tmax_seasonal = [tmax[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
							for year in np.unique(dates.year)
							for season, months in seasons.items()]
			
			tmin_seasonal = [tmin[np.isin(dates.month, months) & (dates.year == year), :, :].mean(axis=0)
							for year in np.unique(dates.year)
							for season, months in seasons.items()]

			pr = np.array(pr_seasonal)
			tmax = np.array(tmax_seasonal)
			tmin = np.array(tmin_seasonal)
			logging.info(f"Aggregated data shape (seasonal): {pr.shape}, {tmax.shape}, {tmin.shape}")

		elif self.config['aggregation'] == 'annual':
			# Group by year using the dates array
			pr_annual = [pr[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]
			tmax_annual = [tmax[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]
			tmin_annual = [tmin[dates.year == year, :, :].mean(axis=0) for year in np.unique(dates.year)]

			pr = np.array(pr_annual)
			tmax = np.array(tmax_annual)
			tmin = np.array(tmin_annual)
			logging.info(f"Aggregated data shape (annual): {pr.shape}, {tmax.shape}, {tmin.shape}")

		return pr, tmax, tmin


	def PRISM(self, start_year=None, end_year=None) -> np.ndarray:
		logging.info("Extracting PRISM data.")
		if start_year is None:
			assert 'start_year' in self.config, "start_year is not provided."
			start_year = self.config['start_year']
		if end_year is None:
			assert 'end_year' in self.config, "end_year is not provided."
			end_year = self.config['end_year']

		mask = self.get_mask()	

		#logging.info(f"Size of the mask: {mask.shape}")
		ppts, tmaxs, tmins = [], [], []
		PRISM_path = '/data/MyDataBase/HydroGeoDataset/PRISM_ML_250m.h5'

		with h5py.File(PRISM_path, 'r') as f:
			logging.info(f"PRISM keys: {f.keys()}")
			logging.info(f"Range of available years: {f['ppt'].keys()}")
			
			for year in range(start_year, end_year+1):	
				logging.info(f"Size of the original PRISM data (year {year}): {f[f'ppt/{year}/data'].shape}")
				logging.info(f"Extracting PRISM data for the year {year}.")

				if self.config['huc8'] is not None:
					row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				else:
					row_max, row_min, col_max, col_min = None, None, None, None

				ppt = f[f'ppt/{year}/data'][:, row_min:row_max, col_min:col_max]
				tmax = f[f'tmax/{year}/data'][:, row_min:row_max, col_min:col_max]
				tmin = f[f'tmin/{year}/data'][:, row_min:row_max, col_min:col_max]

				logging.info(f"Mask shape: {mask.shape}")
				logging.info(f"Size of the PRISM data after cropping (year {year}): {ppt.shape}, {tmax.shape}, {tmin.shape}")

				ppt = np.where(mask != 1, -999, ppt)
				tmax = np.where(mask != 1, -999, tmax)
				tmin = np.where(mask != 1, -999, tmin)

				ppts.append(ppt)
				tmaxs.append(tmax)
				tmins.append(tmin)
		print(f"ppt shape: {ppt.shape}, tmax shape: {tmax.shape}, tmin shape: {tmin.shape}")
		# Concatenate along the first axis to form 3D arrays
		ppts = np.concatenate(ppts, axis=0)
		tmaxs = np.concatenate(tmaxs, axis=0)
		tmins = np.concatenate(tmins, axis=0)

		if self.config['video']:
			self.video_data(ppts, 'ppt_PRISM')	
			self.video_data(tmaxs, 'tmax_PRISM')
			self.video_data(tmins, 'tmin_PRISM')
		
		## replace nan with -999
		ppts = np.where(np.isnan(ppts), -999, ppts)
		tmaxs = np.where(np.isnan(tmaxs), -999, tmaxs)
		tmins = np.where(np.isnan(tmins), -999, tmins)
			
		logging.info(f"loaded PRISM data shape: {ppts.shape}, {tmaxs.shape}, {tmins.shape}")
		
		if self.config['aggregation']:
			ppts, tmaxs, tmins = self.aggregate_temporal_data(ppts, tmaxs, tmins)
			logging.info(f"Aggregated data shape: {ppts.shape}, {tmaxs.shape}, {tmins.shape}")

		return ppts, tmaxs, tmins


	def plot_feature(self, array2d, array_name, categorical=False) -> None:
		logging.info(f"Plotting {array_name} data.")
		plt.figure()
		array2d = np.where(array2d == -999, np.nan, array2d)
		array2d = np.where(array2d == 55537, np.nan, array2d)

		if categorical:
			unique_values = np.unique(array2d[~np.isnan(array2d)])  # Get unique non-NaN values
			num_classes = len(unique_values)

			# Generate random colors
			colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(num_classes)]
			cmap = ListedColormap(colors)
			bounds = np.arange(num_classes + 1) - 0.5
			norm = BoundaryNorm(bounds, cmap.N)
		else:
			cmap = 'viridis'
			norm = None
			vmin = np.nanpercentile(array2d, 1)
			vmax = np.nanpercentile(array2d, 99)

		if categorical:
			plt.imshow(array2d, cmap=cmap, norm=norm)
		else:
			plt.imshow(array2d, vmin=vmin, vmax=vmax, cmap=cmap)

		plt.colorbar()
		plt.title(array_name)
		plt.xlabel("Column")
		plt.ylabel("Row")
		os.makedirs("input_figs", exist_ok=True)
		plt.savefig(f"input_figs/{array_name}.png")
		plt.close()

	def apply_numerical_scale(self, array2d, array_name):
		logging.info(f"###Numerical {array_name}: shape: {array2d.shape}")
		if self.config['plot']: self.plot_feature(np.where(array2d < 0, np.nan, array2d), array_name)
		array2d = np.where(array2d < 0, -999, array2d)
		return array2d

	def apply_categorical_encoding(self, array2d, array_name):
		logging.info(f"###Categorical {array_name}: shape: array2d.shape")
		encoder = LabelEncoder()
		array2d = np.array([encoder.fit_transform(column) for column in array2d.T]).T
		if self.config['plot']: self.plot_feature(array2d, array_name, categorical=True)
		return array2d

	def get_mask(self):
		with h5py.File(self.config['database_path'], 'r') as f:
			DEM_ = f[f"geospatial/BaseRaster_{self.config['RESOLUTION']}m"][:]
			if self.config['huc8'] is not None:
				row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				DEM_ = DEM_[row_min:row_max, col_min:col_max]
			mask = np.where(DEM_ == -999, 0, 1)
			if self.config['plot']: self.plot_feature(mask, "mask_domain")
			logging.info(f"Mask shape: {mask.shape}")
			return mask
	

	def extract_features(self, input_path=None, pattern=None, ET=None, single_location=None) -> gpd.GeoDataFrame:
		"""
		Summary:
		Extract features from a GeoDataFrame based on the nearest neighbor search using KDTree.

		Explanation:
		This function reads a GeoDataFrame from the input path, converts it to EPSG 4326, and extracts features based on the nearest neighbor search using KDTree. It replaces -999 values with NaN, filters out NaN values, and appends the extracted features to the original GeoDataFrame.

		Args:
			self: The instance of the class.
			input_path (str): The path to the input GeoDataFrame file.

		Returns:
			gdf (GeoDataFrame): The GeoDataFrame with extracted features added as new columns.
		"""
		if single_location:
			
			### the single location is (lat, lon)
			### create a dataframe using that lat,lon
			from shapely.geometry import Point
			gdf = pd.DataFrame({'geometry': [Point(single_location[1], single_location[0])]})
			### convert to geodataframe
			gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

		else:
			### if the file is picke, use pandas
			if input_path.endswith('.pkl'):
				gdf = pd.read_pickle(input_path).to_crs(epsg=4326)	
			elif input_path.endswith('.geojson'):
				gdf = gpd.read_file(input_path, driver='GeoJSON').to_crs(epsg=4326)
			else:
				gdf = gpd.read_file(input_path).to_crs(epsg=4326)

		print(f"loaded gdf shape: {gdf.shape}")
		with h5py.File(self.config['database_path'], 'r') as f:
			lat_ = f[f"geospatial/lat_{self.config['RESOLUTION']}m"][:]
			lon_ = f[f"geospatial/lon_{self.config['RESOLUTION']}m"][:]

			# Replace -999 with nan
			lat_ = np.where(lat_ == -999, np.nan, lat_)
			lon_ = np.where(lon_ == -999, np.nan, lon_)
			print(f"loaded lat shape: {lat_.shape}, lon shape: {lon_.shape}")
			# Valid mask to filter out nan values
			valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
			valid_lat = lat_[valid_mask]
			valid_lon = lon_[valid_mask]
			coordinates = np.column_stack((valid_lat, valid_lon))

			# Build KDTree for efficient nearest neighbor search
			tree = cKDTree(coordinates)
			print(f"Keys in the h5 file: {f.keys()}")
			# Extract features with the same RESOLUTION and valid mask
			#features = [feature for feature in f.keys() if f[feature].shape == lat_.shape]
			### get all features in the h5 file
			features = [feature for feature in f.keys() if hasattr(f[feature], 'shape') and f[feature].shape == lat_.shape]
			if pattern:
				
				## always include DEM_250m in the features
				features = [feature for feature in features if pattern in feature]
				features.append(f"DEM_{self.config['RESOLUTION']}m")

			if ET:
				## get all dataset in group "MODIS_ET"
				ET_features = [feature for feature in f["MODIS_ET"].keys() if hasattr(f["MODIS_ET"][feature], 'shape') and f["MODIS_ET"][feature].shape == lat_.shape]
				### add group name to the ET_features
				ET_features = [f"MODIS_ET/{feature}" for feature in ET_features]
				features.extend(ET_features)

				
			# Initialize a dictionary to hold the new columns
			print(f"features: {features}")
			import time
			time.sleep(10)
			feature_data = {feature: [] for feature in features}

			lat_min, lat_max = np.nanmin(lat_), np.nanmax(lat_)
			lon_min, lon_max = np.nanmin(lon_), np.nanmax(lon_)
			
			for lat, lon in zip(gdf.geometry.y, gdf.geometry.x):
				# Check if lat and lon are within the range
				if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
					# Query the nearest neighbor
					distance, index = tree.query([lat, lon])
					logging.info(f" Extracting features for {lat:.2f}, {lon:.2f} with distance {distance}")

					# Get the indices in the original arrays
					lat_index, lon_index = np.where(valid_mask)[0][index], np.where(valid_mask)[1][index]
					
					for feature in features:
						# Collect the feature value
						feature_data[feature].append(f[feature][lat_index, lon_index])
				else:
					# Assign -999 if the point is out of range
					for feature in features:
						feature_data[feature].append(-999)

			# Create a new DataFrame from the collected feature data
			new_feature_df = pd.DataFrame(feature_data)

			# Concatenate the new features with the original GeoDataFrame
			gdf = pd.concat([gdf, new_feature_df], axis=1)
			logging.info(f"Extracted features: {features}")
			return gdf






	def import_pfas_data(self):
		
		if self.config['huc8']:
			row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
		else:
			row_min, row_max, col_min, col_max = 0, -1, 0, -1

		logging.info(f"Reading PFAS data from {self.config['pfas_database_path']}")

		
		with h5py.File(self.config['pfas_database_path'], 'r') as f_pfas:
			print(f_pfas['Max'].keys())
			pfas_max = np.array(f_pfas[f"/Max/{self.config['PFAS']}.tif"][:][row_min:row_max, col_min:col_max])
			pfas_mean = np.array(f_pfas[f"/Mean/{self.config['PFAS']}.tif"][:][row_min:row_max, col_min:col_max])
			pfas_std = np.array(f_pfas[f"/Std/{self.config['PFAS']}.tif"][:][row_min:row_max, col_min:col_max])

		try:
			logging.info("PFAS max shape: %s, range: %s - %s", pfas_max.shape, np.max(pfas_max), np.min(pfas_max[pfas_max != -999]))
			logging.info("PFAS mean shape: %s, range: %s - %s", pfas_mean.shape, np.max(pfas_mean), np.min(pfas_mean[pfas_mean != -999]))
			logging.info("PFAS std shape: %s, range: %s - %s", pfas_std.shape, np.max(pfas_std), np.min(pfas_std[pfas_std != -999]))
		except Exception:
			logging.info("NO PFAS DATA for %s", self.config['PFAS'])

		return pfas_max, pfas_mean, pfas_std

	def import_static_data(self, huc8=True) -> np.ndarray:# numerical_data, categorical_data, groups
		logging.info("Importing static data")
		if self.config['huc8'] and huc8:
			row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
		else:
			row_min, row_max, col_min, col_max = 0, -1, 0, -1

		with h5py.File(self.config['database_path'], 'r') as f:

			numerical_data = [
				self.apply_numerical_scale(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)
				for array_name in self.get_var_name("numerical", self.config['RESOLUTION'], self.config)
			]

			# Extract categorical data
			categorical_data = []
			for array_name in self.get_var_name("categorical", self.config['RESOLUTION'], self.config):
				print(f"array_name: {array_name}")
				print(f"array shape: {f.keys()}")
				array_data = np.array(f[array_name][:][row_min:row_max, col_min:col_max])
				print(f"array_data shape: {array_data.shape}")
				encoded_data = self.apply_categorical_encoding(array_data, array_name)
				categorical_data.append(encoded_data)

			groups = self.apply_categorical_encoding(np.array(f[f"geospatial/COUNTY_{self.config['RESOLUTION']}m"][:][row_min:row_max, col_min:col_max]), f"COUNTY_{self.config['RESOLUTION']}m")
		### change the shape of the numerical data from tuple to numpy array
		numerical_data = np.array(numerical_data)
		categorical_data = np.array(categorical_data)

		logging.info(f"Numerical data shape: {numerical_data.shape}")
		logging.info(f"Categorical data shape: {categorical_data.shape}")
		logging.info(f"Groups data shape: {groups.shape}")

		return numerical_data, categorical_data, groups

	def import_transient_data(self, time_range, name) -> np.ndarray: # name: ppt or recharge

		if self.config['huc8']:
			row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
		else:
			row_min, row_max, col_min, col_max = 0, -1, 0, -1

		with h5py.File(self.config['database_path'], 'r') as f:
			all_data = []
			for year in time_range:
				data = f[f'{name}_{year}_{self.config["RESOLUTION"]}m'][:-1, :-1]
				if self.config['huc8']:
					data = data[row_min:row_max, col_min:col_max]
				all_data.append(data)

		all_data = np.array(all_data)
		logging.info(f"Size of the {name} data: {all_data.shape}")
		for i, year in enumerate(time_range):
			if self.config['plot']: self.plot_feature(all_data[i], f"{name}_{year}")
		return all_data
	
if __name__ == "__main__":
	config = {
			"RESOLUTION": 250,
			
			"geospatial": True,
		}

	importer = DataImporter(config)

	gw_station_data = importer.gw_stations_ds(start_year=1990, end_year=2021)

	print(f"numerical feature: {gw_station_data['421332085401901_1609_389']['numerical_feature']}")
	print(f"categorical feature: {gw_station_data['421332085401901_1609_389']['categorical_feature']}")
	print(f"head: {gw_station_data['421332085401901_1609_389']}")

