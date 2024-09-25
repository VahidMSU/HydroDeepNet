import os
import h5py
import numpy as np
import rasterio


def create_batch_size_dataset(reference_raster, h5_path, size):
    # Ensure files exist
    assert os.path.exists(reference_raster), "Reference raster not found"
    assert os.path.exists(h5_path), "H5 file not found"

    # Open the reference raster and read the array
    with rasterio.open(reference_raster) as src:
        reference_array = src.read(1)
        no_value = src.nodata

    # Create an array with the same shape as the reference array
    rows, cols = reference_array.shape
    batch_size = size

    # Initialize a zero array to hold the batch labels (set uninitialized regions to -999)
    batch_array = np.full_like(reference_array, -999, dtype=np.int32)

    # Label the cells with unique values, ensuring batches are exactly sizexsize unless they are on the edges
    batch_label = 1
    for i in range(0, rows, batch_size):
        for j in range(0, cols, batch_size):
            # Check if the region is exactly sizexsize
            if i + batch_size <= rows and j + batch_size <= cols:
                batch_array[i:i+batch_size, j:j+batch_size] = batch_label
                batch_label += 1
            else:
                # Keep the label as -999 for non-sizexsize regions (edge cases)
                print(f"Skipping edge region at ({i}, {j}) - not sizexsize")

    # Print the number of unique values (excluding -999 for edge regions)
    unique_batches = np.unique(batch_array)
    print(f"Number of unique batch regions (excluding -999): {len(unique_batches[unique_batches != -999])}")

    # Open the HDF5 file and write the batch array as a dataset
    with h5py.File(h5_path, "a") as f:
        # If dataset exists, delete it first
        if f"{size}_{size}_batch_size" in f:
            del f[f"{size}_{size}_batch_size"]
        f.create_dataset(f"{size}_{size}_batch_size", data=batch_array, dtype="int32")
        print(f"Dataset f'{size}_{size}_batch_size' created in {h5_path}")

    # Open the HDF5 file and verify the dataset
    with h5py.File(h5_path, "r") as f:
        assert f"{size}_{size}_batch_size" in f, "Dataset not found"
        data = f[f"{size}_{size}_batch_size"][:]
        print(f"Dataset shape: {data.shape}")
        print(f"Unique values (including -999): {np.unique(data)}")
        print(f"Dataset type: {data.dtype}")
        print(f"Dataset max: {data.max()}")
        print(f"Dataset min: {data.min()}")



if __name__ == "__main__":
    # Define the paths
# Define paths for reference raster and H5 file
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    h5_path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"
    size = 512
    # Create the batch size dataset
    create_batch_size_dataset(reference_raster, h5_path, size)