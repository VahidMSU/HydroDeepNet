import geopandas as gpd
import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from multiprocessing import Pool
from shapely.geometry import box

RESOLUTION = 30

# Load GeoDataFrame
path = f'/data/MyDataBase/SWATGenXAppData/observations/rasters_{RESOLUTION}m_with_observations.pk1'
gdf = gpd.GeoDataFrame(pd.read_pickle(path), crs='EPSG:26990', geometry='geometry')

with rasterio.open(f'/data/MyDataBase/SWATGenXAppData/all_rasters/HUC8_{RESOLUTION}m.tif') as src:
	reference_transform = src.transform
	reference_crs = src.crs
	reference_nodata = src.nodatavals[0]
	reference_width = src.width
	reference_height = src.height
	reference_bounds = src.bounds

# Filter out geometries that are outside the reference raster extent
gdf = gdf[gdf.geometry.intersects(box(*reference_bounds))]

# Create a spatial index
sindex = gdf.sindex

# Function to rasterize a single column
def rasterize_column(column):
	print(f'Creating raster for {column}...')
	output_path = f'/data/MyDataBase/SWATGenXAppData/all_rasters/obs_{column}_{RESOLUTION}m_v2.tif'
	# Prepare shapes and values for rasterization
	shapes_and_values = [(geom, value) for geom, value in zip(gdf.geometry, gdf[column]) if pd.notnull(value)]
	# Rasterize
	raster = rasterize(
		shapes=shapes_and_values,
		out_shape=(reference_height, reference_width),
		transform=reference_transform,
		fill=reference_nodata, # Use the reference nodata value to fill the raster
		dtype=np.float32,
	)
	# Write the raster to a file
	with rasterio.open(
		output_path, 'w', driver='GTiff',
		height=reference_height, width=reference_width,
		count=1, dtype=np.float32, crs=reference_crs,
		transform=reference_transform, nodata=reference_nodata
	) as dst:
		dst.write(raster, 1)

# Columns to create rasters for
columns = ['SWL', 'AQ_THK_1', 'AQ_THK_2', 'H_COND_1', 'H_COND_2', 'V_COND_1', 'V_COND_2',
		'TRANSMSV_1', 'TRANSMSV_2',]

# Use a multiprocessing pool to rasterize the columns in parallel
if __name__ == '__main__':
	rasterize_column(columns[0])
#	with Pool() as p:
#		p.map(rasterize_column, columns)