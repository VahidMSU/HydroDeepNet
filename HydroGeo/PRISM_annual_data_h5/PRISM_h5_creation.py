import h5py
import xarray as xr
import os
import itertools

### this script is to convert the PRISM data to HDF5 format
### input: PRISM data in netCDF format
### output: HDF5 format

variables = ["ppt", "tmax", "tmin"]
years = range(1990, 2023)
output_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/PRISM.h5"

# Open the central HDF5 file
with h5py.File(output_path, 'a') as h5file:  # Use 'a' mode to append to the file if it exists
    for variable, year in itertools.product(variables, years):
        # Define the path to the NetCDF file
        path = f"/data/MyDataBase/SWATGenXAppData/PRISM/CONUS/{variable}/{year}.nc"

        # Open the NetCDF dataset using xarray
        data = xr.open_dataset(path)

        # Print the keys (variables) in the dataset
        print(data.keys())

        # Extract the year from the path (assuming the year is part of the file name)
        year = os.path.basename(path).split('.')[0]

        # Write each variable in the dataset to the HDF5 file under the hierarchy variable/year/
        for var in data.data_vars:
            # Get the data as a numpy array
            var_data = data[var].values
            # Check if the dataset already exists
            dataset_name = f"{variable}/{year}"
            if dataset_name in h5file:
                print(f"Dataset {dataset_name} already exists. Skipping...")
            else:
                # Create a dataset in the HDF5 file under the hierarchy variable/year/
                h5file.create_dataset(dataset_name, data=var_data)

        # Write coordinates and attributes only if they don't already exist
        if "coords" not in h5file:
            # Write each coordinate in the dataset to the HDF5 file under the hierarchy coords/coordinate/
            for coord in data.coords:
                # Get the data as a numpy array
                coord_data = data[coord].values
                # Check if the dataset already exists
                coord_name = f"coords/{coord}"
                if coord_name in h5file:
                    print(f"Coordinate dataset {coord_name} already exists. Skipping...")
                else:
                    # Create a dataset in the HDF5 file under the hierarchy coords/coordinate/
                    h5file.create_dataset(coord_name, data=coord_data)

            # Write global attributes to the HDF5 file
            for attr in data.attrs:
                h5file.attrs[attr] = data.attrs[attr]

        print(f"Data for {variable} in {year} successfully written to {output_path}")

print("All data successfully written to the central HDF5 file.")