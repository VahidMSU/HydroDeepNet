import numpy as np
import rasterio
import os 
# Define paths
RESOLUTION = 30
reference_raster_path = fr'/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif'
workspace = r"/data/MyDataBase/SWATGenXAppData/all_rasters"

# Open the reference raster
with rasterio.open(reference_raster_path) as src:
	# Get the metadata
	meta = src.meta

	# Create coordinate arrays
	x_coords = np.arange(meta['width']) * meta['transform'][0] + meta['transform'][2]
	y_coords = np.arange(meta['height']) * meta['transform'][4] + meta['transform'][5]

	# Create X and Y coordinate rasters
	x_coord_raster = np.repeat(x_coords[np.newaxis, :], meta['height'], axis=0)
	y_coord_raster = np.repeat(y_coords[:, np.newaxis], meta['width'], axis=1)

	# Update the metadata
	meta.update(dtype=rasterio.float32)

	# Save the coordinate rasters
	with rasterio.open(os.path.join(workspace, f'x_{RESOLUTION}m.tif'), 'w', **meta) as dst:
		dst.write(x_coord_raster.astype(rasterio.float32), 1)
	with rasterio.open(os.path.join(workspace, f'y_{RESOLUTION}m.tif'), 'w', **meta) as dst:
		dst.write(y_coord_raster.astype(rasterio.float32), 1)