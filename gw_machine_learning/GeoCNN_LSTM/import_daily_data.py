import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

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
