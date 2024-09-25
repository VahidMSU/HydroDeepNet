import pandas as pd
import numpy as np
import geopandas as gpd
import os

NAMES = os.listdir("/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12")
NAMES.remove('log.txt')

for NAME in NAMES:
    path = f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/PRISM"

    files = os.listdir(path)
    files = [f for f in files if f.endswith('.pcp')]

    lats = []
    lons = []
    elevs = []
    names = []

    for file in files:
        with open(f'{path}/' + file, 'r') as f:
            lines = f.readlines()
            lats.append(float(lines[2].split()[2]))
            lons.append(float(lines[2].split()[3]))
            elevs.append(float(lines[2].split()[4]))
            names.append(os.path.basename(file).split('.')[0])

    # Create a DataFrame with latitude, longitude, and elevation
    data = pd.DataFrame({'lat': lats, 'lon': lons, 'elev': elevs, 'name': names})

    # Create a GeoDataFrame from the DataFrame
    geometry = gpd.points_from_xy(data.lon, data.lat)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326').to_crs('EPSG:26990')

    # Save the GeoDataFrame as a shapefile
    output_path = f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/PRISM/PRISM_grid.shp"
    gdf.to_file(output_path)
