#logging.info(f"Getting geometries for HUC12s: {list_of_huc12s}")
### list of huc12 are like:  ['020200030604', '020200030603', '020200030601', '020200030602', '020200030605']
import geopandas as gpd
import pandas as pd
VPUID = "0407"
print(VPUID)
# Read the shapefile
path = f"/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/NHDWaterbody.pkl"
path2 = f"/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/NHDArea.pkl"

gdf = gpd.GeoDataFrame(pd.read_pickle(path)).to_crs("EPSG:4326")
import matplotlib.pyplot as plt 
print(f"columns: {gdf.columns}")
gdf.plot()
plt.savefig(f"lakes.png")
