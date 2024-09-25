import arcpy
import os
import sys
import geopandas as gpd
import multiprocessing
import pandas as pd
from global_tools import rt


	
if __name__ == '__main__':
	lat = "/data/MyDataBase/SWATGenXAppData/Grid/lat_250m.tif"
	lon = "/data/MyDataBase/SWATGenXAppData/Grid/lon_250m.tif"
	overwrite = False
	rt.extract_raster(lat, "/data/MyDataBase/SWATGenXAppData/all_rasters/lat_250m.tif", 250, overwrite)
	rt.extract_raster(lon, "/data/MyDataBase/SWATGenXAppData/all_rasters/lon_250m.tif", 250, overwrite)
	