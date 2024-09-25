from calendar import c
from datetime import date
from logging import config
import h5py
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import random
import matplotlib.animation as animation
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

class DataImporter:
	def __init__(self, config, device=None):
		"""
		Summary:
		Class for importing and processing various types of data for hydrogeological modeling.

		Explanation:
		This class provides methods for importing and processing different types of data including static, transient, and PFAS data. It handles tasks such as extracting features and preparing data for deep learning.

		"""


		self.config = config
		self.device = device
		self.config['database_path'] = f'/data/MyDataBase/HydroGeoDataset_ML_{config["RESOLUTION"]}.h5'
		self.config['geoloc'] = False if 'geoloc' not in config else config['geoloc']
		self.config['snow'] = False if 'snow' not in config else config['snow']
		self.config['groundwater'] = False if 'groundwater' not in config else config['groundwater']
		self.config['population_array'] = False if 'population_array' not in config else config['population_array']
		self.config['landfire'] = False if 'landfire' not in config else config['landfire']
		self.config['geology'] = False if 'geology' not in config else config['geology']
		self.config['NHDPlus'] = False if 'NHDPlus' not in config else config['NHDPlus']
		self.config['plot'] = False if 'plot' not in config else config['plot']
		self.config['pfas_database_path'] = f'/data/MyDataBase/PFAS_sw_{config["RESOLUTION"]}m.h5'
		self.config['huc8'] = None if 'huc8' not in config else config['huc8']
		self.config['snowdas_h5_path'] = '/data/MyDataBase/SNODAS.h5'
		self.config['video'] = False if 'video' not in config else config['video']

	def get_var_name(self, feature_type, RESOLUTION, config):
		features = []
		if feature_type == 'categorical':
			features = [
				f'COUNTY_{self.config["RESOLUTION"]}m',
				f'landforms_{config["RESOLUTION"]}m_250Dis',
				f'gSURRGO_swat_{self.config["RESOLUTION"]}m',
				f'landuse_{self.config["RESOLUTION"]}m',
			]
			if config['geology']:
				features.extend([
					f'geomorphons_{config["RESOLUTION"]}m_250Dis',
					f'MI_geol_poly_{self.config["RESOLUTION"]}m',
					f'Glacial_Landsystems_{self.config["RESOLUTION"]}m',
					f'Aquifer_Characteristics_Of_Glacial_Drift_{self.config["RESOLUTION"]}m',
				])
		elif feature_type == 'numerical':
			self.select_numerical_features(config, features, RESOLUTION)
		return features

	def select_numerical_features(self, config, features, RESOLUTION) -> None:
		if config["geoloc"]:
			features.extend([
				f'DEM_{self.config["RESOLUTION"]}m',
				f'x_{self.config["RESOLUTION"]}m',
				f'y_{self.config["RESOLUTION"]}m',
			])

		if config["snow"]:
			features.extend([
					f'non_snow_accumulation_raster_{self.config["RESOLUTION"]}m',
					f'snow_accumulation_raster_{self.config["RESOLUTION"]}m',
					f'melt_rate_raster_{self.config["RESOLUTION"]}m',
					f'average_temperature_raster_{self.config["RESOLUTION"]}m',
					f'snow_layer_thickness_raster_{self.config["RESOLUTION"]}m',
				])

		if config['groundwater']:
			features.extend([
				f'kriging_output_H_COND_1_{self.config["RESOLUTION"]}m',
				f'kriging_output_AQ_THK_1_{self.config["RESOLUTION"]}m',
				f'kriging_output_H_COND_2_{self.config["RESOLUTION"]}m',
				f'kriging_output_SWL_{self.config["RESOLUTION"]}m',
				f'kriging_output_V_COND_2_{self.config["RESOLUTION"]}m',
				f'kriging_output_TRANSMSV_1_{self.config["RESOLUTION"]}m',
				f'kriging_output_TRANSMSV_2_{self.config["RESOLUTION"]}m',
				f'kriging_output_V_COND_1_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_SWL_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_H_COND_1_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_H_COND_2_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_V_COND_1_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_V_COND_2_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_AQ_THK_1_{self.config["RESOLUTION"]}m',
				f'kriging_stderr_AQ_THK_2_{self.config["RESOLUTION"]}m',
			])

		if config['NHDPlus'] is not None:
			features.extend([
				f'QAMA_MILP_{self.config["RESOLUTION"]}m',        ## mean annual streamflow
				f'QBMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow from excess ET
				f'QCMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow with reference gage regression
				f'QDMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow with NHDPlusAdditionRemoval
				f'QEMA_MILP_{self.config["RESOLUTION"]}m',        ## Mean annual flow from gage adjustment
				f'QIncrBMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow with excess ET
				f'QIncrCMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow by subtracting upstream QCMA
				f'QFMA_MILP_{self.config["RESOLUTION"]}m',
				f'QGAdjMA_MILP_{self.config["RESOLUTION"]}m',
				f'QIncrAMA_MILP_{self.config["RESOLUTION"]}m',
				f'QIncrDMA_MILP_{self.config["RESOLUTION"]}m', 	  ## Incremental flow with NHDPlusAdditionRemoval
				f'QIncrEMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow from gage adjustment
				f'QIncrFMA_MILP_{self.config["RESOLUTION"]}m',    ## Incremental flow from gage sequestration
				f'VBMA_MILP_{self.config["RESOLUTION"]}m',      # Velocity for QBMA
				f'VCMA_MILP_{self.config["RESOLUTION"]}m',		# Velocity for
				f'VDMA_MILP_{self.config["RESOLUTION"]}m',      # Velocity for QCMA
				f'VEMA_MILP_{self.config["RESOLUTION"]}m',		# Velocity from gage adjustment
			])
		if config['landfire']:
			features.extend([
				f'LC20_Asp_220_{self.config["RESOLUTION"]}m',
				f'LC20_BPS_220_{self.config["RESOLUTION"]}m',
				f'LC20_EVT_220_{self.config["RESOLUTION"]}m',
				f'LC20_Elev_220_{self.config["RESOLUTION"]}m',
				f'LC20_SlpD_220_{self.config["RESOLUTION"]}m',
				f'LC20_SlpP_220_{self.config["RESOLUTION"]}m',
				f'LC22_EVC_220_{self.config["RESOLUTION"]}m',
				f'LC22_EVH_220_{self.config["RESOLUTION"]}m',
			])

		if config['population_array'] is not None:
			features.extend([
				f'pden1990_ML_{self.config["RESOLUTION"]}m',
				f'pden2000_ML_{self.config["RESOLUTION"]}m',
				f'pden2010_ML_{self.config["RESOLUTION"]}m',
			])

	def unpack_and_combine(self, numerical_data, categorical_data):
		numerical_tensors = [torch.tensor(nd, dtype=torch.float32) for nd in numerical_data]
		categorical_tensors = [torch.tensor(cd, dtype=torch.long) for cd in categorical_data]
		return torch.stack(numerical_tensors + categorical_tensors, dim=0)


	def gw_3d_ds(self, start_year=2020, end_year=2022) -> np.ndarray:
		""" Extract the 3D groundwater head data from the database. """

		if self.config['RESOLUTION'] != 250:
			raise ValueError("Groundwater head data is only available at 250m resolution.")
		path = '/data/MyDataBase/gw_head.h5'
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
				print("All nan values in the groundwater head data.")
				return None

		print(f"#### Groundwater head shape: {gw_head.shape}")
		return gw_head

	## add hints for the stations
	def gw_stations_ds(self, stations=None, start_year=2020, end_year=2022) -> np.ndarray:
		""" Extract the groundwater head data for specific stations from the database. """

		path = '/data/MyDataBase/gw_head_2d.h5'

		numerical, categorical, _ = importer.import_static_data(huc8=False)

		with h5py.File(path, 'r') as f:
			## NOTE: date from 1-1-1990 to 12-31-2022
			start_index = (start_year - 1990) * 365
			end_index = (end_year - 1990) * 365
			gw_station_data = {}
			#if stations is None:
			stations = f.keys()
			print(f"## stations: {stations}")
			for station in stations:

				gw_head = f[station][start_index:end_index]
				print(f"## station: {station} with shape: {gw_head.shape} and %{100*(1- sum(np.isnan(gw_head)/gw_head.size)):.2f} observations. ")
				row = station.split('_')[1]
				col = station.split('_')[2]
				numerical_feature =  numerical[:, int(row), int(col)]
				categorical_feature = categorical[:, int(row), int(col)]

				## if all nan, return None
				if np.isnan(gw_head).all():
					print(f"All nan values in the groundwater head data for station {station}.")
					continue
				### add the features to the dictionary
				gw_station_data[station] = {
					'gw_head': gw_head,
					'numerical_feature': numerical_feature,
					'categorical_feature': categorical_feature,
				}

		print(f"head length: {gw_station_data['421332085401901_1609_389']['gw_head'].shape}")
		print(f"numerical feature: {gw_station_data['421332085401901_1609_389']['numerical_feature'].shape}")
		print(f"categorical feature: {gw_station_data['421332085401901_1609_389']['categorical_feature'].shape}")


		return gw_station_data

	def tensorize(self, array3d):
		percentile_99 = np.percentile(array3d[array3d != -999], 99)
		array3d = np.where(array3d > percentile_99, -999, array3d)
		array3d = torch.tensor(array3d, dtype=torch.float32)
		return array3d

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
			print(h5_file[f'250m/{year}'].keys())

			var = h5_file[f"250m/{year}/{snowdas_var}"][:]
			unit = h5_file[f"250m/{year}/{snowdas_var}"].attrs['units']
			print(f"Size of the SNOWDAS data: {var.shape}")
			convertor = h5_file[f"250m/{year}/{snowdas_var}"].attrs['converters']
			print(f"Convertor: {convertor}")
			var = np.where(var == 55537, np.nan, var*convertor)
			if self.config['huc8']:

				min_x, max_x, min_y, max_y  = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				var = var[:, min_x:max_x, min_y:max_y]


			print(f"Size of the SNOWDAS data after cropping: {var.shape}")
			if self.config['video']:
				self.video_data(var, f"{snowdas_var}_{unit}_{year}")
		return var
	def video_data(self, data, name) -> None:
		### create video of the data, each 2d array from the 3d array is a frame
		### data: 3d array
		fig, ax = plt.subplots()

		def update_frame(i):
			ax.clear()
			im = ax.imshow(data[i], animated=True)
			ax.set_title(f"Step {i+1}")
			return [im]

		ani = animation.FuncAnimation(fig, update_frame, frames=range(data.shape[0]), interval=50, blit=True, repeat_delay=1000)
		os.makedirs('input_videos', exist_ok=True)
		if self.config['huc8']:
			ani.save(f'input_videos/{name}_{self.config["huc8"]}.gif', writer='pillow')
		else:
			ani.save(f'input_videos/{name}.gif', writer='pillow')
		plt.close(fig)
		print(f"Video of {name} data saved.")

	def LOCA2(self, start=0, end=-1, row = None, col = None):

		""" The function imports the climate data from the database
		and applies the necessary preprocessing steps."""

		path = '/data/MyDataBase/LOCA2_MLP.h5'
		with h5py.File(path, 'r') as f:
			print(f.keys())
			print("reading data")
			if row is not None and col is not None:
				pr = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/pr'][start:end, row, col]
				tmax = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/tasmax'][start:end, row, col]
				tmin = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/tasmin'][start:end, row, col]
				return pr, tmax, tmin
			else:
				pr = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/pr'][start:end]      # 3D array, shape: (23741, 67, 75)
				tmax = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/tasmax'][start:end] # 3D array, shape: (23741, 67, 75)
				tmin = f[f'e_n_cent/{self.config["cc_model"]}/{self.config["scenario"]}/{self.config["ensemble"]}/{self.config["cc_time_step"]}/{self.config["time_range"]}/tasmin'][start:end]
				print("Size of the climate data: ", pr.shape, tmax.shape, tmin.shape)

				if self.config["RESOLUTION"] == 250:
					# Calculate the replication factors
					rep_factors = (int(np.ceil(1848 / pr.shape[1])), int(np.ceil(1457 / pr.shape[2])))
					print(f"Replication factors: {rep_factors}")

					# Replicate the climate data using numpy.repeat
					pr = np.repeat(pr, rep_factors[0], axis=1)
					pr = np.repeat(pr, rep_factors[1], axis=2)

					tmax = np.repeat(tmax, rep_factors[0], axis=1)
					tmax = np.repeat(tmax, rep_factors[1], axis=2)

					tmin = np.repeat(tmin, rep_factors[0], axis=1)
					tmin = np.repeat(tmin, rep_factors[1], axis=2)

					print("Replication completed.")

					# Flip the climate data to correct the orientation
					pr = np.flip(pr, axis=1).copy()
					tmax = np.flip(tmax, axis=1).copy()
					tmin = np.flip(tmin, axis=1).copy()
					print("Flipping completed.")

					# Pad the climate data to achieve the exact target shape
					target_shape = (pr.shape[0], 1848, 1457)
					pr = pr[:, :target_shape[1], :target_shape[2]]
					tmax = tmax[:, :target_shape[1], :target_shape[2]]
					tmin = tmin[:, :target_shape[1], :target_shape[2]]
					print("Padding completed.")

				print("Size of the climate data after replication and padding: ", pr.shape, tmax.shape, tmin.shape)

			if self.config['huc8'] is not None:

				row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])

				pr = pr[:, row_min:row_max, col_min:col_max]
				tmax = tmax[:, row_min:row_max, col_min:col_max]
				tmin = tmin[:, row_min:row_max, col_min:col_max]

			print("Size of the climate data after slicing: ", pr.shape, tmax.shape, tmin.shape)
			if self.config['video']:
				self.video_data(pr, 'pr')
				self.video_data(tmax, 'tmax')
				self.video_data(tmin, 'tmin')

			return pr, tmax, tmin


	def plot_feature(self, array2d, array_name, categorical=False) -> None:
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
		print(f"###Numerical {array_name}:", "shape:", array2d.shape)
		if self.config['plot']: self.plot_feature(np.where(array2d < 0, np.nan, array2d), array_name)
		array2d = np.where(array2d < 0, -999, array2d)
		return array2d

	def apply_categorical_encoding(self, array2d, array_name):
		print(f"###Categorical {array_name}:", "shape:", array2d.shape)
		encoder = LabelEncoder()
		array2d = np.array([encoder.fit_transform(column) for column in array2d.T]).T
		if self.config['plot']: self.plot_feature(array2d, array_name, categorical=True)
		return array2d
	def get_mask(self):
		with h5py.File(self.config['database_path'], 'r') as f:
			DEM_ = f[f"BaseRaster_{self.config['RESOLUTION']}m"][:]
			if self.config['huc8'] is not None:
				row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
				DEM_ = DEM_[row_min:row_max, col_min:col_max]
			mask = np.where(DEM_ == -999, 0, 1)
			if self.config['plot']: self.plot_feature(mask, "mask_domain")
			print(f"Mask shape: {mask.shape}")
			return mask

	def extract_features(self, input_path) -> gpd.GeoDataFrame:
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

		gdf = gpd.read_file(input_path).to_crs(epsg=4326)

		with h5py.File(self.config['database_path'], 'r') as f:
			lat_ = f[f"lat_{self.config['RESOLUTION']}m"][:]
			lon_ = f[f"lon_{self.config['RESOLUTION']}m"][:]

			# Replace -999 with nan
			lat_ = np.where(lat_ == -999, np.nan, lat_)
			lon_ = np.where(lon_ == -999, np.nan, lon_)

			# Valid mask to filter out nan values
			valid_mask = ~np.isnan(lat_) & ~np.isnan(lon_)
			valid_lat = lat_[valid_mask]
			valid_lon = lon_[valid_mask]
			coordinates = np.column_stack((valid_lat, valid_lon))

			# Build KDTree for efficient nearest neighbor search
			tree = cKDTree(coordinates)

			# Extract features with the same resolution and valid mask
			features = [feature for feature in f.keys() if f[feature].shape == lat_.shape]

			# Initialize a dictionary to hold the new columns
			feature_data = {feature: [] for feature in features}

			lat_min, lat_max = np.nanmin(lat_), np.nanmax(lat_)
			lon_min, lon_max = np.nanmin(lon_), np.nanmax(lon_)

			for lat, lon in zip(gdf.geometry.y, gdf.geometry.x):
				# Check if lat and lon are within the range
				if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
					# Query the nearest neighbor
					distance, index = tree.query([lat, lon])
					print(f" Extracting features for {lat:.2f}, {lon:.2f} with distance {distance}")

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
			print(f"Extracted features: {features}")
			return gdf

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

		with h5py.File(database_path, 'r') as f:
			huc8 = np.array(f[f'HUC8_{self.config["RESOLUTION"]}m'][:])

		rows, cols = np.where(huc8 == int(huc8_select))
		row_min, row_max = rows.min(), rows.max()
		col_min, col_max = cols.min(), cols.max()
		return row_min, row_max, col_min, col_max


	def import_pfas_data(self):
		if self.config['huc8']:
			row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
		else:
			row_min, row_max, col_min, col_max = 0, -1, 0, -1
		print(f"Reading PFAS data from {self.config['pfas_database_path']}")

		with h5py.File(self.config['pfas_database_path'], 'r') as f_pfas:
			pfas_max = np.array(f_pfas[f"/Max/{self.config['PFAS']}"][:][row_min:row_max, col_min:col_max])
			pfas_mean = np.array(f_pfas[f"/Mean/{self.config['PFAS']}"][:][row_min:row_max, col_min:col_max])
			pfas_std = np.array(f_pfas[f"/Std/{self.config['PFAS']}"][:][row_min:row_max, col_min:col_max])
		try:
			print(f"PFAS max shape: {pfas_max.shape}", "range:", np.max(pfas_max), np.min(pfas_max[pfas_max != -999]))

			print(f"PFAS mean shape: {pfas_mean.shape}", "range:", np.max(pfas_mean), np.min(pfas_mean[pfas_mean != -999]))

			print(f"PFAS std shape: {pfas_std.shape}", "range:", np.max(pfas_std), np.min(pfas_std[pfas_std != -999]))
		except Exception:
			print(f"NO PFAS DATA for {self.config['PFAS']}")

		return pfas_max, pfas_mean, pfas_std
	def import_static_data(self, huc8=True) -> np.ndarray:# numerical_data, categorical_data, groups

		if self.config['huc8'] and huc8:
			row_min, row_max, col_min, col_max = self.get_huc8_ranges(self.config['database_path'], self.config['huc8'])
		else:
			row_min, row_max, col_min, col_max = 0, -1, 0, -1

		with h5py.File(self.config['database_path'], 'r') as f:

			numerical_data = [
				self.apply_numerical_scale(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)
				for array_name in self.get_var_name("numerical", self.config['RESOLUTION'], self.config)
			]

			categorical_data = [
				self.apply_categorical_encoding(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)
				for array_name in self.get_var_name("categorical", self.config['RESOLUTION'], self.config)
			]

			groups = self.apply_categorical_encoding(np.array(f[f"COUNTY_{self.config['RESOLUTION']}m"][:][row_min:row_max, col_min:col_max]), f"COUNTY_{self.config['RESOLUTION']}m")
		### change the shape of the numerical data from tuple to numpy array
		numerical_data = np.array(numerical_data)
		categorical_data = np.array(categorical_data)

		print(f'######## shape of the numerical data: {numerical_data.shape}')
		print(f'######## shape of the categorical data: {categorical_data.shape}')
		print(f'######## shape of the groups data: {groups.shape}')

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
		print(f"Size of the {name} data: ", all_data.shape)
		for i, year in enumerate(time_range):
			if self.config['plot']: self.plot_feature(all_data[i], f"{name}_{year}")
		return all_data

	def recharge_rainfall_ds(self) -> torch.Tensor: # combined_input_train, target_tensor_train, combined_input_test, target_tensor_test, dataset_train, mask
		numerical_data, categorical_data, _ = self.import_static_data()

		train_years = range(self.config['start_training_year'], self.config['end_training_year'] + 1)
		test_years = range(self.config['start_testing_year'], self.config['end_testing_year'] + 1)

		rainfall_train = self.import_transient_data(train_years, "ppt")
		recharge_train = self.import_transient_data(train_years, "recharge")
		rainfall_test = self.import_transient_data(test_years, "ppt")
		recharge_test = self.import_transient_data(test_years, "recharge")

		static_tensor = self.unpack_and_combine(numerical_data, categorical_data).to(self.device)
		static_tensor_train = static_tensor.unsqueeze(0).expand(len(train_years), -1, -1, -1)
		static_tensor_test = static_tensor.unsqueeze(0).expand(len(test_years), -1, -1, -1)

		rainfall_tensor_train = self.tensorize(rainfall_train).to(self.device).unsqueeze(1)
		recharge_tensor_train = self.tensorize(recharge_train).to(self.device).unsqueeze(1)
		rainfall_tensor_test = self.tensorize(rainfall_test).to(self.device).unsqueeze(1)
		recharge_tensor_test = self.tensorize(recharge_test).to(self.device).unsqueeze(1)

		combined_input_train = torch.cat((static_tensor_train, rainfall_tensor_train), dim=1).unsqueeze(0)
		target_tensor_train = recharge_tensor_train.unsqueeze(0).to(self.device)
		combined_input_test = torch.cat((static_tensor_test, rainfall_tensor_test), dim=1).unsqueeze(0)
		target_tensor_test = recharge_tensor_test.unsqueeze(0).to(self.device)

		dataset_train = TensorDataset(combined_input_train, target_tensor_train)

		print(f"Combined input shape (train): {combined_input_train.shape}")
		print(f"Target tensor shape (train): {target_tensor_train.shape}")
		print(f"Combined input shape (test): {combined_input_test.shape}")
		print(f"Target tensor shape (test): {target_tensor_test.shape}")

		mask = self.get_mask()
		mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

		return combined_input_train, target_tensor_train, combined_input_test, target_tensor_test, dataset_train, mask


if __name__ == '__main__':

	config = {
		"RESOLUTION": 250,
		"huc8": "4060105",
		"PFAS": "PFOS",
		"cc_model": "ACCESS-CM2",
		"scenario": "historical",
		"ensemble": "r2i1p1f1",
		"time_range": "1950_2014",
		"cc_time_step": 'daily',
      	'snow': True,
        'geoloc': False,
        'groundwater': False,
        'population_array': True,
        'landfire': False,
        'geology': False,
        'NHDPlus': False,
        'plot': False,
		'start_training_year': 2004,
		'end_training_year': 2012,
		'start_testing_year': 2013,
		'end_testing_year': 2019,
		'video': True,
	}
	### import CNN-LSTM training and testing dataset
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#importer = DataImporter(config, device)
	#combined_input_train, target_tensor_train, combined_input_test, target_tensor_test, dataset_train, mask = importer.recharge_rainfall_ds()

	# import climate change data for a specific time period and 250m resolution

	#importer = DataImporter(config)
	#pr, tmax, tmin = importer.LOCA2(start=1,end=20)  ## kg m-2 s-1, K, K

	#numerical, categorical, groups = importer.import_static_data()


	# get mask
	#mask = importer.get_mask()


	# Import annual precipitation PRISM data for a specific time period
	#annual_rainfall = importer.import_transient_data(range(1990, 2022), "ppt")   ## mm
	#average_daily_recharge =importer.import_transient_data(range(1990, 2022), "recharge")  ## mm/day

	# import numerical and categorical datasets
	#import_static_data = importer.import_static_data()
	# import PFAS data #
	#config = {
	#	"RESOLUTION": 30,
    #	"PFAS": "PFOS"}   ### othr PFAS: PFHxS, PFOA, PFNA, PFDA, PFOS
	#pfas_max, pfas_mean, pfas_std = importer.import_pfas_data()

	##########################################################################
	# extract information for point locations
	#config = {
	#	"RESOLUTION": 250}

	input_path = "/data/MyDataBase/Huron_River_gw_PFAS/PFAS_gw_data.geojason"
	output_path = "/data/MyDataBase/Huron_River_gw_PFAS/PFAS_gw_data_with_features.pkl"
	config = {
		"RESOLUTION": 250}
	importer = DataImporter(config)
	gdf = importer.extract_features(input_path)
	gdf.to_pickle(output_path)
	gdf.to_file(output_path.replace(".pkl", ".geojson"), driver='GeoJSON')

	###########################################################################

	# Print the first 5 rows
	#print(gdf.head())
	#print(f"number of rows: {len(gdf)}")
	# Plot the shapefile
	#gdf.plot()
	#plt.savefig("input_figs/P_locations_rasters_30m.png")

	# import snowdas data
	#melt_rate = importer.extract_snowdas_data(snowdas_var='snow_layer_thickness', year = 2015)  #'melt_rate', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate'. data range from 2004 to 2019


	#config = {
	#	"RESOLUTION": 250,
	#	"huc8": "4060105",
		#"video": True,
	#}
	#importer = DataImporter(config)


	#gw_3d_ds = importer.gw_3d_ds(start_year=2020, end_year=2021)

	#gw_station_data = importer.gw_stations_ds(start_year=1990, end_year=2021)
	# print detail of one station
	# print the features of the station
	#print(f"numerical feature: {gw_station_data['421332085401901_1609_389']['numerical_feature']}")
	#print(f"categorical feature: {gw_station_data['421332085401901_1609_389']['categorical_feature']}")
	# print(f"head: {gw_station_data['421332085401901_1609_389']}")

	# import training and testing dataset for 3D CNN-LSTM
