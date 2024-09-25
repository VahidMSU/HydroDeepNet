import arcpy
import geopandas
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from get_extent import get_lat_lon_reference_raster
import h5py
import numpy as np
import os

snowdas = "/data/MyDataBase/SWATGenXAppData/snow/"
## list of all the files in the directory ending with nc
files = [f for f in os.listdir(snowdas) if f.endswith(".nc")]
## print the files
print(files)


min_lon, max_lon, min_lat, max_lat = get_lat_lon_reference_raster("/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif")
print(min_lon, max_lon, min_lat, max_lat)


path = "/data/MyDataBase/SWATGenXAppData/snow/SNODAS_Modeled_melt_rate_constrained_2004_2023.nc"

ds = xr.open_dataset(path)
## print shape and size of the dataset

print(ds.coords)

## limit the dataset to the extent of the reference raster
ds = ds.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))

## save in a h5 file
ds.to_netcdf("/data/MyDataBase/SWATGenXAppData/snow/SNODAS_Modeled_melt_rate_constrained_2004_2023.h5")
