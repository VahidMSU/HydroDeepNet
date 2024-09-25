import h5py
import numpy as np

with h5py.File("/data/MyDataBase/HydroGeoDataset_ML_250.h5", 'r') as f:
    print(f.keys())
    BaseRaster_250m = f['BaseRaster_250m'][:]
    print(np.unique(BaseRaster_250m))   