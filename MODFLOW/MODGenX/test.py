import os 

path = "/data/SWATGenXApp/GenXAppData/observations/observations.geojson"

import geopandas as gpd

gdf = gpd.read_file(path)

print(f"colums: {gdf.columns}")