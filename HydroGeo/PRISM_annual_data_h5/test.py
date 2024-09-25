import h5py
import numpy as np
import matplotlib.pyplot as plt

# Define variables and years
variables = ["ppt", "tmax", "tmin"]
years = range(1990, 2023)
output_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/PRISM.h5"

# Open the central HDF5 file
with h5py.File(output_path, 'r') as f:  # Use 'r' mode to read the file
	print("Opening the central HDF5 file")
	print(f'Keys: {f.keys()}')
	lats = f['coords/lat'][:]
	lons = f['coords/lon'][:]
	print(f"Latitude range: {lats.min()} to {lats.max()}")
	print(f"Longitude range: {lons.min()} to {lons.max()}")
	# shape
	print(f"Shape of the latitude array: {lats.shape}")
	print(f"Shape of the longitude array: {lons.shape}")
	rows = f['coords/row'][:]	
	cols = f['coords/col'][:]
	print(f"Row range: {rows.min()} to {rows.max()}")
	print(f"Column range: {cols.min()} to {cols.max()}")
	# shape
	print(f"Shape of the row array: {rows.shape}")
	print(f"Shape of the column array: {cols.shape}")

	ppt_2010 = f['ppt/2010/data']
	print(f"Shape of the sliced array: {ppt_2010.shape}")
	## plot
	plt.imshow(ppt_2010[0, :, :])
	plt.colorbar()
	plt.show()
