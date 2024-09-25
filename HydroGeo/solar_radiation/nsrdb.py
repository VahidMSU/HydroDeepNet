import h5py
import pandas as pd
import numpy as np
import geopandas as gpd
from netCDF4 import Dataset

# Load PRISM grid
prism_conus = pd.read_pickle('/data/MyDataBase/SWATGenXAppData/PRISM/prism_4km_mesh/prism_4km_mesh.pkl')
prism_conus = gpd.GeoDataFrame(prism_conus, geometry='geometry', crs='EPSG:4326')
extent = prism_conus.total_bounds

# Function to filter coordinates
def filter_coordinates(coordinates, extent):
    return np.where(
        (coordinates[:, 0] >= extent[0]) & (coordinates[:, 0] <= extent[2]) &
        (coordinates[:, 1] >= extent[1]) & (coordinates[:, 1] <= extent[3])
    )[0]

# Processing and resampling function
def process_and_resample(file_path, filtered_indices):
    with h5py.File(file_path, 'r') as f:
        # Extract relevant data
        time_index = pd.to_datetime(f['time_index'][:].astype(str))
        ghi_data = f['ghi'][:, filtered_indices] * f['ghi'].attrs['psm_scale_factor']
        ghi_unit = f['ghi'].attrs['psm_units']
        
        # Resample to daily sums
        date_series = pd.Series(time_index.date)
        unique_dates = date_series.unique()
        daily_ghi = np.zeros((len(unique_dates), ghi_data.shape[1]))
        
        for i, date in enumerate(unique_dates):
            daily_indices = np.where(date_series == date)[0]
            daily_ghi[i, :] = ghi_data[daily_indices, :].sum(axis=0)
    
    return unique_dates, daily_ghi, ghi_unit

# Main processing function
def main():
    file_path = 'E:/NSRDB/nsrdb_2017_full.h5'
    with h5py.File(file_path, 'r') as f:
        coordinates = f['coordinates'][:]
        filtered_indices = filter_coordinates(coordinates, extent)
    
    unique_dates, daily_ghi, ghi_unit = process_and_resample(file_path, filtered_indices)
    
    # Save to NetCDF
    nc_filename = 'E:/NSRDB/nsrdb_2017_daily.nc'
    with Dataset(nc_filename, 'w', format='NETCDF4') as nc:
        # Define dimensions
        nc.createDimension('time', None)
        nc.createDimension('location', daily_ghi.shape[1])

        # Create variables
        times = nc.createVariable('time', 'f4', ('time',))
        locations = nc.createVariable('location', 'i4', ('location',))
        ghi = nc.createVariable('ghi', 'f4', ('time', 'location',), zlib=True)
        ghi.units = ghi_unit
        # Write data
        times[:] = np.arange(len(unique_dates))
        locations[:] = filtered_indices
        ghi[:, :] = daily_ghi
        # Add global attributes
        nc.description = 'Daily aggregated GHI values filtered by PRISM extent'

if __name__ == '__main__':
    main()
