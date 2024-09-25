import geopandas as gpd
import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
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

# Columns to create rasters for
columns = ['SWL', 'AQ_THK_1', 'AQ_THK_2', 'H_COND_1', 'H_COND_2', 'V_COND_1', 'V_COND_2',
		'TRANSMSV_1', 'TRANSMSV_2',]
for column in columns:
	print(f'Creating raster for {column}...')
	output_path = f'/data/MyDataBase/SWATGenXAppData/all_rasters/obs_{column}_{RESOLUTION}m.tif'
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
