import h5py
import xarray as xr
import os 
h5_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI_resampled_250m"

# Open the HDF5 file in read mode
print(os.listdir(h5_path))
