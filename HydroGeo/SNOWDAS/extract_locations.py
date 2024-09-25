
########################### creating location shapeifles

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np


def extract_lat_lon_for_shapefile(nc_file):
    # Open the NetCDF file
    dataset = nc.Dataset(nc_file, 'r')
    
    # Extract latitude and longitude data
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    
    dataset.close()
    
    return lons, lats

nc_file = 'D:\MyDataBase\snow\SNODAS_melting.nc'
lats, lons = extract_lat_lon_for_shapefile(nc_file)                    
lats_shape = lats.shape  # Assuming lats is a 2D NumPy array
lons_shape = lons.shape  # Assuming lons is a 2D NumPy array

# Flatten the arrays and also get the corresponding row and column indices
lat_flat = lats.flatten()
long_flat = lons.flatten()
row_indices, col_indices = np.indices(lats_shape)

# Create DataFrame
SNODAS_stations = pd.DataFrame({
    'geometry': [Point(x, y) for x, y in zip(long_flat, lat_flat)],
    'lat': lat_flat,
    'long': long_flat,
    'row_id': row_indices.flatten(),
    'col_id': col_indices.flatten(),
    'ID': np.arange(100000, 100000 + lat_flat.size)
})

# Convert to GeoDataFrame
SNODAS_stations = gpd.GeoDataFrame(SNODAS_stations, crs='EPSG:4326', geometry='geometry')


# Save to shapefile
SNODAS_stations.to_file(fr'D:\MyDataBase\snow\SNODAS_locations.shp')