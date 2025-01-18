import rasterio 
import numpy as np
import os
import xarray as xr
import calendar
import geopandas as gpd
import pandas as pd
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths


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
TYPES = ['ppt']  # ['ppt', 'tmax', 'tmin']

for TYPE in TYPES:
    print(f"Processing {TYPE}...")
    for YEAR in YEARS:
        arrays = []
        for MONTH in MONTHS:
            _, last_day = calendar.monthrange(YEAR, MONTH)
            for DAY in range(1, last_day + 1):
                extract_path = f"{SWATGenXPaths.PRISM_path}/unzipped_daily/{TYPE}/{YEAR}/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil/PRISM_{TYPE}_stable_4kmD2_{YEAR}{MONTH:02}{DAY:02}_bil.bil"
                # now create the NetCDF file using all of the bil files for a type
                with rasterio.open(extract_path) as src:
                    array = src.read(1)
                    

                arrays.append(array)

        # Now we have all of the arrays for a year and a type, we can stack them together
        stacked = np.stack(arrays)

        # Now we can create the NetCDF file
        print(f"Creating the NetCDF file for {TYPE}... with the shape of {stacked.shape}")
        path = f"{SWATGenXPaths.PRISM_path}/CONUS/{TYPE}/{YEAR}.nc"
        os.makedirs(os.path.dirname(path), exist_ok=True)

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
        ds.to_netcdf(path)

print("Processing complete.")
