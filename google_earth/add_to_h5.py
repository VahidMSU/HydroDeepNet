import h5py
import numpy as np
import os 
import rasterio



def add_to_h5(h5_path, list_of_rasters, h5_group_name="MODIS_ET"):
    # Open the HDF5 file
    with h5py.File(h5_path, 'a') as f:
        # Create the datasets
        ### delete the group if already exists
        if h5_group_name in f:
            del f[h5_group_name]
            
        f.create_group(h5_group_name)


        for raster in list_of_rasters:
            name = raster.replace('.tif', '')
            print(f"Processing {name}")
            with rasterio.open(f"downloads/{raster}") as src:
                img = src.read(1)
                no_value = src.nodata
                ## replace no value with -999
                img = np.where(img == no_value, -999, img)
                ## if already exists, delete it

                f.create_dataset(f"{h5_group_name}/{name}", data=img, dtype=np.float32)
                print(f"Shape of the dataset: {f[h5_group_name][name].shape}")
                print(f"Dataset created: {name}")
            print(f"Dataset created: {name}")

    print("All datasets created successfully!")

if __name__ == "__main__":
    h5_path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"
    assert os.path.exists(h5_path), "The HDF5 file does not exist!"

    list_of_rasters = os.listdir("downloads")
    # ending with .tif
    list_of_rasters = [raster for raster in list_of_rasters if raster.endswith(".tif")]

    add_to_h5(h5_path, list_of_rasters)



