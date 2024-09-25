import xarray as xr
import h5py
year = 2000
variable = "ppt"
path = f"/data/MyDataBase/SWATGenXAppData/PRISM/CONUS/{variable}/{year}.nc"
## read the coordinates
data = xr.open_dataset(path)
lat = data["lat"].values
lon = data["lon"].values
print(lat)
print(lon)