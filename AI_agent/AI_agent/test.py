import h5py
import numpy as np
import rasterio
import os
from osgeo import gdal

usle_k_250m = "/data2/MyDataBase/SWATGenXAppData/all_rasters/usle_k_swat_aggregated_250m.tif"
reference_raster = "/data2/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"

# Paths to the HDF5 files
path1 = "/data/SWATGenXApp/GenXAppData/HydroGeoDataset/HydroGeoDataset_ML_250.h5"
path2 = "/data2/MyDataBase/HydroGeoDataset_ML_250.h5"

with h5py.File(path1, 'r') as f:
    data = f["geospatial/lat_250m"][:]
    print(data.shape)