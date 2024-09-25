import os
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def get_lat_lon(cc_file):
    # Open the dataset
    ds = xr.open_dataset(cc_file)
    # Print keys
    print(f"keys: {ds.keys()}")
    lat = ds['lat'].values
    lon = ds['lon'].values
    print(f"lat shape: {lat.shape}")
    print(f"lon shape: {lon.shape}")
    # Print max min
    print(f"lat max: {np.max(lat)}", f"lat min: {np.min(lat)}")
    print(f"lon max: {np.max(lon)}", f"lon min: {np.min(lon)}")
    return lat, lon

if __name__ == "__main__":
    # Lat lon for great lake region
    scenario = 'ssp585'
    ensemble = 'r1i1p1f2'
    resolution = '0p0625deg'
    model = 'CNRM-CM6-1-HR'
    region = 'e_n_cent'
    variable = 'pr'

    current_dir = os.path.dirname(os.path.realpath(__file__))
    cc_path = f"/data/MyDataBase/SWATGenXAppData/climate_change/cirrus.ucsd.edu/~pierce/LOCA2/CONUS_regions_split/{model}/{region}/{resolution}/{ensemble}/{scenario}/{variable}/"


    # Find the nc file
    cc_file = None
    files = os.listdir(cc_path)
    for file in files:
        if file.endswith('.nc') and "month" not in file:
            print(file)
            cc_file = os.path.join(cc_path, file)
            break
    if cc_file := os.path.join(
        cc_path,
        "pr.CNRM-CM6-1-HR.ssp585.r1i1p1f2.2075-2100.LOCA_16thdeg_v20220519.monthly.e_n_cent.nc",
    ):
        lat, lon = get_lat_lon(cc_file)
        if lat is not None and lon is not None:
            print(f"Number of lat: {len(lat)}")
            print(f"Number of lon: {len(lon)}")

            # Convert longitudes from 0-360 to -180 to 180
            lon = np.where(lon > 180, lon - 360, lon)

            # Generate all possible combinations of latitudes and longitudes
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            lon_flat = lon_grid.flatten()
            lat_flat = lat_grid.flatten()

            # Create a shapefile of lat and lon
            geometry = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
            gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
            gdf.to_file(os.path.join("/data/MyDataBase/SWATGenXAppData/climate_change", f"{region}_grid.shp"))
            print(gdf.head())
        else:
            print("Failed to extract latitude and longitude.")
    else:
        print("No suitable NetCDF file found.")
