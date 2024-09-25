import h5py
import os
import rasterio
import numpy as np

path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/compounds/masked_rasters"
for res in ['30','50','100','250']:
	# Get all the rasters ending with .tif and containing _250_ in the name
	rasters = [file for file in os.listdir(path) if file.endswith(".tif") and f"{res}_" in file]
	print(rasters)

	hdf5_path = f"/data/MyDataBase/SWATGenXAppData/PFAS/PFAS_sw_{res}m.h5"
	## delete if exists
	if os.path.exists(hdf5_path):
		os.remove(hdf5_path)
	else:
		## make sure directory exists
		os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
	# Open the HDF5 file in append mode
	with h5py.File(hdf5_path, 'a') as f:
		for raster in rasters:
			parts = raster.split("_")
			
			res = parts[0]
			stat = parts[1]
			compound = parts[3].split(".")[0]
			
			# Read the raster
			raster_path = os.path.join(path, raster)
			with rasterio.open(raster_path) as src:
				data = src.read(1)
				data = data.astype('float32')  # Ensure data type is float32
				data[data == src.nodata] = -999  # Replace no-data values with -999
				rows, cols = data.shape
				
				# Define the dataset path in HDF5
				dataset_path = f"{stat}/{compound}"
				
				# Check if the dataset already exists
				if dataset_path in f:
					del f[dataset_path]  # Remove existing dataset if it exists
				
				# Create a new dataset
				dataset = f.create_dataset(dataset_path, (rows, cols), dtype='float32')
				dataset[...] = data
				print(f"Dataset {stat}_{compound} has been created in {dataset_path}")
