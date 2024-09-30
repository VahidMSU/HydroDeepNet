import netCDF4 as nc
import matplotlib.pyplot as plt

path = "/data/LOCA2/CONUS_regions_split/ACCESS-CM2/e_n_cent/0p0625deg/r1i1p1f1/historical/pr/pr.ACCESS-CM2.historical.r1i1p1f1.1950-2014.LOCA_16thdeg_v20240915.e_n_cent.monthly.nc"

ds = nc.Dataset(path)
print(ds)
### dimensions
print(ds.dimensions)
## lat, lon max/min
latitudes = ds.variables['lat'][:]
longitudes = ds.variables['lon'][:]
print(f"max lat: {latitudes.max()}, min lat: {latitudes.min()}")
print(f"max lon: {longitudes.max()}, min lon: {longitudes.min()}")

## plot for one month
plt.imshow(ds.variables['pr_tavg'][0, :, :])
plt.colorbar()
plt.savefig('/home/rafieiva/MyDataBase/codebase/HydroGeo/PRIMS_VS_LOCA2/figs/loca2.png')