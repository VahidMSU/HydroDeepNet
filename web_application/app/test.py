import sys
sys.path.append('/data/SWATGenXApp/codes/SWATGenX')
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import geopandas as gpd

FPS_geometry_name_shp_path = SWATGenXPaths.FPS_CONUS_stations
gdf = gpd.read_file(FPS_geometry_name_shp_path)
print(gdf.columns)
