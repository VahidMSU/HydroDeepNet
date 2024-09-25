import arcpy
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from pyproj import CRS

## this script is to convert the PRISM data to raster format
## input: PRISM data in netCDF format
## output: raster format



# Define the EPSG code for the geographic coordinate system
epsg_code = 4269  # EPSG code for GCS_North_American_1983

variable = 'ppt'
year = 1990
for year in range(1990, 2023):
	path = f"/data/MyDataBase/SWATGenXAppData/PRISM/CONUS/{variable}/{year}.nc"
	ds = xr.open_dataset(path)
	print(ds.keys())

	data = ds['data']
	lat = ds['lat']  # Latitudes of the grid cells
	lon = ds['lon']  # Longitudes of the grid cells

	# Replace -9999 with NaN
	data = data.where(data != -9999)

	# Calculate the total annual precipitation
	total_annual = data.sum(dim='time')

	# Fill NaN values with -9999
	total_annual = total_annual.fillna(-9999)
	# Replace 0 with -9999
	total_annual = total_annual.where(total_annual != 0, -9999)

	# Define the output raster path
	output_raster = f"/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/{variable}_{year}.tif"

	# Ensure the arcpy environment is set up
	arcpy.env.overwriteOutput = True
	arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(epsg_code)

	# Convert the total_annual DataArray to a NumPy array
	total_annual_np = total_annual.values

	# Metadata for the raster
	nrows = 621
	ncols = 1405
	ulxmap = -125
	ulymap = 49.9166666666664
	xdim = 0.0416666666667
	ydim = 0.0416666666667
	nodata_value = -9999

	# Define the lower left corner of the raster
	lower_left_x = ulxmap
	lower_left_y = ulymap - (nrows * ydim)

	# Create the raster from the NumPy array
	raster = arcpy.NumPyArrayToRaster(total_annual_np, 
									lower_left_corner=arcpy.Point(lower_left_x, lower_left_y), 
									x_cell_size=xdim, 
									y_cell_size=ydim, 
									value_to_nodata=nodata_value)

	# Set the spatial reference using EPSG code
	spatial_reference = arcpy.SpatialReference(epsg_code)
	arcpy.DefineProjection_management(raster, spatial_reference)

	# Save the raster
	raster.save(output_raster)

	print(f"Raster saved to {output_raster}")
