import rasterio
import numpy as np
import os
import rioxarray
import xarray as xr
import calendar
import geopandas as gpd
import pandas as pd
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths


def is_valid_netcdf(file_path):
    """Check if a NetCDF file exists and is valid"""
    if not os.path.exists(file_path):
        return False

    try:
        # Try to open the file
        with xr.open_dataset(file_path) as ds:
            # Check if it has the required variables and dimensions
            if 'data' not in ds.variables:
                return False

            # Check if the data is not empty
            if ds['data'].size == 0:
                return False

            # Check if all values are not NaN
            if np.all(np.isnan(ds['data'])):
                return False

            # Check if the dimensions are correct
            if len(ds['data'].dims) != 3:  # time, row, col
                return False

            return True
    except Exception as e:
        print(f"Error checking NetCDF file {file_path}: {e}")
        return False


PRISM_path = SWATGenXPaths.PRISM_path
PRISM_mesh_path = SWATGenXPaths.PRISM_mesh_path
PRISM_mesh_pickle_path = SWATGenXPaths.PRISM_mesh_pickle_path

if not os.path.exists(PRISM_mesh_pickle_path):
    PRISM_mesh = gpd.read_file(PRISM_mesh_path).to_crs("EPSG:4326")
    # turn it into a pickle file for future use
    PRISM_mesh.to_pickle(PRISM_mesh_pickle_path)
else:
    PRISM_mesh = pd.read_pickle(PRISM_mesh_pickle_path)

print("Reading the PRISM mesh...Done")

YEARS = np.arange(1990, 2023)
MONTHS = np.arange(1, 12 + 1)
TYPES = ['ppt', 'tmax', 'tmin']

for TYPE in TYPES:
    print(f"Processing {TYPE}...")
    for YEAR in YEARS:
        # Check if a valid NetCDF file already exists
        nc_path = f"{SWATGenXPaths.PRISM_path}/CONUS/{TYPE}/{YEAR}.nc"
        if is_valid_netcdf(nc_path):
            print(f"Valid NetCDF file already exists for {TYPE} {YEAR}, skipping...")
            continue

        arrays = []
        missing_days = []
        for MONTH in MONTHS:
            _, last_day = calendar.monthrange(YEAR, MONTH)
            for DAY in range(1, last_day + 1):
                extract_path = f"{SWATGenXPaths.PRISM_unzipped_path}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil"
                bil_file = f"{extract_path}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.bil"
                hdr_file = f"{extract_path}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.hdr"

                if not os.path.exists(bil_file) or not os.path.exists(hdr_file):
                    missing_days.append(f"{YEAR}-{MONTH:02}-{DAY:02}")
                    continue

                if os.path.getsize(bil_file) == 0 or os.path.getsize(hdr_file) == 0:
                    print(f"Empty files for {YEAR}-{MONTH:02}-{DAY:02}, skipping...")
                    missing_days.append(f"{YEAR}-{MONTH:02}-{DAY:02}")
                    continue

                try:
                    with rasterio.open(bil_file) as src:
                        array = src.read(1)
                        if array.size == 0 or np.all(np.isnan(array)):
                            print(f"Invalid data in {bil_file}, skipping...")
                            missing_days.append(f"{YEAR}-{MONTH:02}-{DAY:02}")
                            continue
                        arrays.append(array)
                except Exception as e:
                    print(f"Error reading {bil_file}: {e}")
                    missing_days.append(f"{YEAR}-{MONTH:02}-{DAY:02}")
                    continue

        if not arrays:
            print(f"No valid data for {TYPE} {YEAR}, skipping...")
            continue

        if missing_days:
            print(f"Missing or invalid data for {len(missing_days)} days in {YEAR}:")
            print("\n".join(missing_days))

        # Now we have all of the arrays for a year and a type, we can stack them together
        stacked = np.stack(arrays)
        # Now we can create the NetCDF file
        print(f"Creating the NetCDF file for {TYPE}... with the shape of {stacked.shape}")
        os.makedirs(os.path.dirname(nc_path), exist_ok=True)

        # Create a new NetCDF file
        ds = xr.Dataset(
            {
                'data': (('time', 'row', 'col'), stacked)
            },
            coords={
                'time': np.arange(stacked.shape[0]),
                'row': np.arange(stacked.shape[1]),
                'col': np.arange(stacked.shape[2])
            }
        )

        # Set the spatial resolution and extent
        ds.rio.write_crs("EPSG:4326", inplace=True)
        ds.rio.set_spatial_dims(x_dim='col', y_dim='row', inplace=True)

        # Write the dataset to a NetCDF file
        ds.to_netcdf(nc_path)

        # Verify the created file
        if not is_valid_netcdf(nc_path):
            print(f"Warning: Created NetCDF file for {TYPE} {YEAR} may be invalid")
            if os.path.exists(nc_path):
                os.remove(nc_path)

print("Processing complete.")