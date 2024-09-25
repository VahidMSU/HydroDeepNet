import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from scipy.stats import shapiro, anderson

report_stat = "/home/rafieiva/MyDataBase/codes/GeoNet/report/feature_stat.txt"
def apply_numerical_scale(array2d, array_name, RESOLUTION):
	if "recharge" in array_name:
		array2d = np.where(array2d != -999, array2d * 365, array2d)
	# Create a mask of valid values (not equal to -999)
	if f"x_{RESOLUTION}m" in array_name:
		array2d = np.where(array2d != -999, array2d / 1000, array2d)

	if f"y_{RESOLUTION}m" in array_name:
		array2d = np.where(array2d != -999, array2d / 1000, array2d)

	# Replace outliers with -999
	if "obs_" in array_name:
		percentile_99 = np.percentile(array2d[array2d != -999], 99.5)
		percentile_1 = np.percentile(array2d[array2d != -999], 0.5)
		array2d = np.where(array2d > percentile_99, -999, array2d)
		array2d = np.where(array2d < percentile_1, -999, array2d)

	# Apply a standard scaler to the 2D array
	# Log transfer of the data except for no values
	if "snow" in array_name or "melt" in array_name or "temperature" in array_name or "thickness" in array_name:
		#valid_values = array2d[array2d != -999]
		#average = np.mean(valid_values)
		#std = np.std(valid_values)
		#print(f"###Numerical {array_name}: average: {average}, std: {std}, shape: {array2d.shape}")
		array2d = np.where((array2d > 0) & np.isfinite(array2d), np.log10(np.maximum(array2d, 1e-10)), -999)

	# Log transfer of the NHD database
	if "MILP" in array_name:
		array2d = np.where((array2d > 0) & np.isfinite(array2d), np.log10(np.maximum(array2d, 1e-10)), -999)

	return array2d

def apply_categorical_encoding(array2d, array_name):
	
	# Apply a label encoder to each column in the 2D array
	print(f"###Categorical {array_name}:", np.unique(array2d), "shape:", array2d.shape)
	with open(report_stat, "a") as f:
		f.write(f"### categorical: {np.unique(array2d)} shape: {array2d.shape}\n")
	encoder = LabelEncoder()
	array2d = np.array([encoder.fit_transform(column) for column in array2d.T]).T
	
	
	return array2d





def get_huc8_ranges(config):
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
	import time
	print(f"### get_huc8_ranges: {config['huc8']}")	

	with h5py.File(config['database_path'], 'r') as f:
		huc8 = np.array(f[f'HUC8_{config["RESOLUTION"]}m'][:])

	rows, cols = np.where(huc8 == int(config['huc8']))
	row_min, row_max = rows.min(), rows.max()
	col_min, col_max = cols.min(), cols.max()
	return row_min, row_max, col_min, col_max

def import_simulated_data(database_path, target_array, numerical_arrays, categorical_arrays, num_samples,RESOLUTION, huc8, config, return_grid=False):

	with h5py.File(database_path, 'r') as f:
		target = apply_numerical_scale(np.array(f[target_array][:]), "TARGET VAR", RESOLUTION)

		numerical_data = [
			apply_numerical_scale(np.array(f[array_name][:]), array_name, RESOLUTION)
			for array_name in numerical_arrays
		]

		categorical_data = [
			apply_categorical_encoding(np.array(f[array_name][:]), array_name)
			
			for array_name in categorical_arrays
		]
		counties = apply_categorical_encoding(np.array(f[f'/COUNTY_{RESOLUTION}m']), f"COUNTY_{RESOLUTION}m")

		if huc8:
			row_min, row_max, col_min, col_max = get_huc8_ranges(config)
			target = target[row_min:row_max, col_min:col_max]
			numerical_data = [array[row_min:row_max, col_min:col_max] for array in numerical_data]
			categorical_data = [array[row_min:row_max, col_min:col_max] for array in categorical_data]
			counties = counties[row_min:row_max, col_min:col_max]


	# Flatten the data if not returning grid format
	if not return_grid:

		numerical_data_flat = [array.flatten() for array in numerical_data]
		categorical_data_flat = [array.flatten() for array in categorical_data]
		target_flat = target.flatten()
		counties_flat = counties.flatten()
		valid_indices = target_flat != -999

		counties_valid = counties_flat[valid_indices]
		numerical_data_valid = [array[valid_indices].reshape(-1, 1) for array in numerical_data_flat]
		categorical_data_valid = [array[valid_indices].reshape(-1, 1) for array in categorical_data_flat]
		
		target_valid = target_flat[valid_indices]
		
		# Ensure we do not exceed the number of valid samples
		min_num_samples = min(len(target_valid), num_samples) if num_samples > 0 else len(target_valid)

		# Combine numerical and categorical data if both exist
		if numerical_data_valid and categorical_data_valid:
			data_valid = np.concatenate(numerical_data_valid + categorical_data_valid, axis=1)
		elif numerical_data_valid:
			data_valid = np.concatenate(numerical_data_valid, axis=1)
		elif categorical_data_valid:
			data_valid = np.concatenate(categorical_data_valid, axis=1)
		
		data_valid = data_valid[:min_num_samples]
		target_valid = target_valid[:min_num_samples]
		counties_valid = counties_valid[:min_num_samples]
	else:

		# In grid format, ensure each array is at least 2D before concatenation
		numerical_data = [array[:, :, np.newaxis] for array in numerical_data]
		categorical_data = [array[:, :, np.newaxis] for array in categorical_data]
		data_valid = np.concatenate(numerical_data + categorical_data, axis=2)
		target_valid = target
		counties_valid = counties

	return data_valid, target_valid, counties_valid