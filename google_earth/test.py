import rasterio
import matplotlib.pyplot as plt
import h5py
with h5py.File("/data/MyDataBase/HydroGeoDataset_ML_250.h5", 'r') as f:
    print(f.keys())
    print(f['MODIS_ET'].keys())
    print(f['MODIS_ET']['MODIS_ET_2023-01-01_to_2023-01-31'].shape)
    img = f['MODIS_ET']['MODIS_ET_2023-01-01_to_2023-01-31'][:]
    plt.imshow(img)
    plt.colorbar()
    plt.savefig("test.png")