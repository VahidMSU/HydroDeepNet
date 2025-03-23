import os
import geopandas as gpd
import numpy as np

path = "/data/SWATGenXApp/GenXAppData/observations/observations.geojson"

gdf = gpd.read_file(path)
print(f"crs: {gdf.crs}")
print("Columns for observations.geojson:")
print(gdf.columns)
print("\nData types for observations.geojson:")
print(gdf.dtypes)
print(f"range of SWL: {np.ptp(gdf['SWL'])}")


path2 = "/data/SWATGenXApp/GenXAppData/observations/observations_original.geojson"

gdf2 = gpd.read_file(path2)

### drop all rows with SWL being NaN
gdf2 = gdf2.dropna(subset=['SWL'])
### required columns:


print(f"crs: {gdf2.crs}")
print("\nColumns for Grids_MODFLOW.geojson:")
print(gdf2.columns)
print("\nData types for Grids_MODFLOW.geojson:")
print(gdf2.dtypes)
print(f"range of SWL: {np.ptp(gdf2['SWL'])}")
