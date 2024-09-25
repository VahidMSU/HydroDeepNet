import arcpy
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt

# Load the counties shapefile
counties = gpd.read_file("/data/MyDataBase/SWATGenXAppData/COUNTY/Counties_(v17a).shp")
print(counties.columns)

# Filter the counties and keep the 'OBJECTID' field
counties = counties[counties['PENINSULA'] == "lower"][['OBJECTID', 'geometry']]

# Save the filtered counties to a temporary shapefile
temp_address = "/data/MyDataBase/SWATGenXAppData/temp/counties.shp"

# Ensure the temp folder exists
if not os.path.exists("/data/MyDataBase/SWATGenXAppData/temp"):
    os.makedirs("/data/MyDataBase/SWATGenXAppData/temp")

counties.to_file(temp_address)

# Reference raster
reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"

arcpy.env.workspace = "/data/MyDataBase/SWATGenXAppData/all_rasters"
arcpy.env.extent = reference_raster
arcpy.env.snapRaster = reference_raster
arcpy.env.cellSize = reference_raster
arcpy.env.outputCoordinateSystem = reference_raster
arcpy.env.overwriteOutput = True
### use -999 for no values
# Rasterize the temporary shapefile
arcpy.PolygonToRaster_conversion(temp_address, "OBJECTID", "/data/MyDataBase/SWATGenXAppData/all_rasters/mask_250m.tif", cellsize=250)
