
import os 
import geopandas as gpd
path = "/data/SWATGenXApp/GenXAppData/observations/observations.geojson"

data = gpd.read_file(path)

print(f"columns: {data.columns}")   

path = "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/MODFLOW_250m/Grids_MODFLOW.geojson"
data = gpd.read_file(path)

print(f"columns: {data.columns}")
