
import xarray as xr
import numpy as np

nc_orig = "/data/LOCA2/CONUS_regions_split/ACCESS-ESM1-5/e_n_cent/0p0625deg/r2i1p1f1/historical/pr/pr.ACCESS-ESM1-5.historical.r2i1p1f1.1950-2014.LOCA_16thdeg_v20220519.e_n_cent.nc"

ds = xr.open_dataset(nc_orig)
print(ds.lon.values)
lat = 45.0
lon = -90.0

orig_lat = ds.lat.values
orig_lon = ds.lon.values
orig_data = ds.pr.values

