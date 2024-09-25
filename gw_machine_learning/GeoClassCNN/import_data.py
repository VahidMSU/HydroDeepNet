import array
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
report_stat = "/home/rafieiva/MyDataBase/codes/gw_machine_learning/report/feature_stat.txt"
def plot_feature(array2d, array_name):
	plt.figure()
	array2d = np.where(array2d == -999, np.nan, array2d)
	plt.imshow(array2d, vmin = np.nanpercentile(array2d, 1), vmax = np.nanpercentile(array2d, 99), cmap='jet')
	plt.colorbar()
	plt.title(array_name)
	plt.xlabel("Column")
	plt.ylabel("Row")
	os.makedirs("input_figs", exist_ok=True)	
	plt.savefig(f"input_figs/{array_name}.png")
	plt.close()
def apply_numerical_scale(array2d, array_name):
	# Create a mask of valid values (not equal to -999)
	print(f"###Numerical {array_name}:", "shape:", array2d.shape)
	
	if "kriging" in array_name:
		## first make sure they are int32
		array2d = array2d.astype(int)
		# Replace negative values with -999
		array2d = np.where(array2d < 0, -999, array2d)
	if "obs_" in array_name:
		array2d = array2d.astype(int)
		array2d = np.where(array2d < 0, -999, array2d)

	if "TARGET VAR" in array_name:
		#array2d = np.where(array2d != -999, ((array2d/(250*25))*100)*356 , array2d)
		## remove outliers 
		percentile_99 = np.percentile(array2d[array2d != -999], 99)
		print(f"target parameter 99th percentile: {percentile_99}")
		array2d = np.where(array2d > percentile_99, -999, array2d)

	plot_feature(array2d, array_name)
	return array2d


def get_mask(database_path, RESOLUTION):
	with h5py.File(database_path, 'r') as f:
		DEM_ = f[f'BaseRaster_{RESOLUTION}m'][:]
		return DEM_ != -999

def apply_categorical_encoding(array2d, array_name):
	print(f"###Categorical {array_name}:", "shape:", array2d.shape)
	# Apply a label encoder to each column in the 2D array
	print(f"###Categorical {array_name}:", np.unique(array2d), "shape:", array2d.shape)
	with open(report_stat, "a") as f:
		f.write(f"### categorical: {np.unique(array2d)} shape: {array2d.shape}\n")
	encoder = LabelEncoder()
	array2d = np.array([encoder.fit_transform(column) for column in array2d.T]).T
	
	
	return array2d


def get_huc8_ranges(database_path, RESOLUTION, huc8_select=False):
	with h5py.File(database_path, 'r') as f:
		huc8 = np.array(f[f'HUC8_{RESOLUTION}m'][:])  # 2D array
	
	rows, cols = np.where(huc8 == int(huc8_select))
	row_min, row_max = rows.min(), rows.max()
	col_min, col_max = cols.min(), cols.max()

	print(row_min, row_max, col_min, col_max)
	### the range should be padded to fit 800*800
	
	return row_min, row_max, col_min, col_max

def import_simulated_data(database_path, target_array, numerical_arrays, categorical_arrays, RESOLUTION, huc8=None, pfas_database_path=None):
	
	""" The function imports the data from the database and applies the necessary preprocessing steps.
		The data that we load are 2d arrays representing 
		a gird with cell size of 250m*250m and 1849*1458 rows and columns."""
	
	### read 
	if huc8:
		row_min, row_max, col_min, col_max = get_huc8_ranges(database_path, RESOLUTION, huc8)
	else:
		row_min, row_max, col_min, col_max = 0, -1, 0, -1

	with h5py.File(database_path, 'r') as f:
		if pfas_database_path:
			with h5py.File(pfas_database_path, 'r') as f_pfas:
				## get the pfas data
				pfas = np.array(f_pfas["/Max/PFOS"][:][row_min:row_max, col_min:col_max ])
		if target_array:
			target = apply_numerical_scale(np.array(f[target_array][:][row_min:row_max, col_min:col_max ]), "TARGET VAR")
		else:
			target = None
		numerical_data = [
			apply_numerical_scale(np.array(f[array_name][:][row_min:row_max, col_min:col_max]), array_name)
			for array_name in numerical_arrays
		]

		categorical_data = [
			apply_categorical_encoding(np.array(f[array_name][:][row_min:row_max, col_min:col_max ]), array_name)
			
			for array_name in categorical_arrays
		]

		groups = apply_categorical_encoding(np.array(f[f"COUNTY_{RESOLUTION}m"][:][row_min:row_max, col_min:col_max ]), f"COUNTY_{RESOLUTION}m")
	
	return target, numerical_data, categorical_data, groups

def LOCA2(RESOLUTION, database_path, huc8=None):


	""" The function imports the climate data from the database 
	and applies the necessary preprocessing steps."""

	path = '/data/climate_change/LOCA2_MLP.h5'
	with h5py.File(path, 'r') as f:
		print(f.keys())
		print("reading data")
		pr = f['e_n_cent/ACCESS-CM2/historical/r2i1p1f1/daily/1950_2014/pr'][:100]      # 3D array, shape: (23741, 67, 75)
		tmax = f['e_n_cent/ACCESS-CM2/historical/r2i1p1f1/daily/1950_2014/tasmax'][:100] # 3D array, shape: (23741, 67, 75)
		tmin = f['e_n_cent/ACCESS-CM2/historical/r2i1p1f1/daily/1950_2014/tasmin'][:100] 
		print("Size of the climate data: ", pr.shape, tmax.shape, tmin.shape)
		
		if RESOLUTION == 250:
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

		plt.imshow(pr[0])
		plt.savefig("pr.png")
		
		print("Size of the climate data after replication and padding: ", pr.shape, tmax.shape, tmin.shape)

	### read 
	if huc8:
		row_min, row_max, col_min, col_max = get_huc8_ranges(database_path, RESOLUTION, huc8)
		pr = pr[:, row_min:row_max, col_min:col_max]
		tmax = tmax[:, row_min:row_max, col_min:col_max]
		tmin = tmin[:, row_min:row_max, col_min:col_max]


	


	print("Size of the climate data after slicing: ", pr.shape, tmax.shape, tmin.shape)
	plt.imshow(pr[0])
	plt.savefig("pr.png")
	return pr, tmax, tmin





def import_recharge_rainfall_data(RESOLUTION, database_path, time_range, huc8=None):
	
	with h5py.File(database_path, 'r') as f:
		all_ppts = []
		all_recharges = []
		for year in time_range:
			ppt = f[f'ppt_{year}_{RESOLUTION}m'][:-1, :-1]
			recharge = f[f'recharge_{year}_{RESOLUTION}m'][:-1, :-1]

			if huc8:
				row_min, row_max, col_min, col_max = get_huc8_ranges(database_path, RESOLUTION, huc8)
				ppt = ppt[row_min:row_max, col_min:col_max]
				recharge = recharge[row_min:row_max, col_min:col_max]
			all_ppts.append(ppt)
			all_recharges.append(recharge)
	## convert to 3d numpy
	all_ppts = np.array(all_ppts)
	all_recharges = np.array(all_recharges)
	print("Size of the ppt data: ", all_ppts.shape)
	print("Size of the recharge data: ", all_recharges.shape)
	for i in range(all_ppts.shape[0]):
		plot_feature(all_ppts[i], f"ppt_{i}")
		plot_feature(all_recharges[i], f"recharge_{i}")
	return all_ppts, all_recharges