import xarray as xr
import matplotlib.pyplot as plt

prims_path = "/data/PRISM/CONUS/ppt/1990.nc"
prism = xr.open_dataset(prims_path)
print(prism)