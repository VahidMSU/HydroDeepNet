import h5py
import numpy as np

# Define the path to the central HDF5 file
output_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/PRISM.h5"
years = range(1990, 2023)

# Function to list all datasets in the HDF5 file
def list_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name)

# Open the HDF5 file in read mode
with h5py.File(output_path, 'r') as h5file:
    # List all datasets
    print("Datasets in the HDF5 file:")
    h5file.visititems(list_datasets)

    # Verify data for a specific variable and year
    variable = "ppt"
    year = "1990"
    
    # Check if the dataset exists
    dataset_name = f"{variable}/{year}"
    if dataset_name in h5file:
        data = h5file[dataset_name][:]
        print(f"\nData for {variable} in {year}:")
        print(data)
    else:
        print(f"Dataset {dataset_name} not found in the HDF5 file.")

    # Verify coordinates
    coord = "lat"
    coord_name = f"coords/{coord}"
    if coord_name in h5file:
        coord_data = h5file[coord_name][:]
        print(f"\nCoordinate data for {coord}:")
        print(coord_data)
    else:
        print(f"Coordinate dataset {coord_name} not found in the HDF5 file.")

    coord = "lon"
    coord_name = f"coords/{coord}"
    if coord_name in h5file:
        coord_data = h5file[coord_name][:]
        print(f"\nCoordinate data for {coord}:")
        print(coord_data)
    else:
        print(f"Coordinate dataset {coord_name} not found in the HDF5 file.")
        
    # Extract a time series for one row_range and col_rangeumn (e.g., row_range 0 and col_rangeumn 0)
    lat, lon = (40, 47), (-90, -80)
    lat_idx = np.where((h5file["coords/lat"][:] >= lat[0]) & (h5file["coords/lat"][:] <= lat[1]))
    lon_idx = np.where((h5file["coords/lon"][:] >= lon[0]) & (h5file["coords/lon"][:] <= lon[1]))
    
    if len(lat_idx[0]) > 0 and len(lon_idx[0]) > 0:
        row_range = (lat_idx[0][0], lat_idx[0][-1])  # Fix indexing
        col_range = (lon_idx[0][0], lon_idx[0][-1])  # Fix indexing

        print(f"Row range: {row_range}, Column range: {col_range}")

        timeseries = []
        for year in years:
            dataset_name = f"{variable}/{year}"
            if dataset_name in h5file:
                data = h5file[dataset_name][:]
                timeseries.append(data[:, row_range[0]:row_range[1]+1, col_range[0]:col_range[1]+1])  # Fix indexing
            else:
                print(f"Dataset {dataset_name} not found in the HDF5 file.")

        # Combine the time series data into a single array
        if timeseries:
            timeseries = np.concatenate(timeseries, axis=0)
            print(f"\nTime series for {variable} at row range {row_range}, column range {col_range} from 1990 to 2022:")
            print(timeseries)
        else:
            print("No time series data found for the specified variable and years.")
    else:
        print("No data found for the specified latitude and longitude range.")
