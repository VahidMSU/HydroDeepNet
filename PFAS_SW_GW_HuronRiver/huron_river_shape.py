import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_River_basin_bound.pkl"
gdf = pd.read_pickle(path)
print(gdf.head())
gdf.plot()
plt.show()

## save as shapefile
gdf.to_file("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_River_basin_bound.shp")
## save as geojson
gdf.to_file("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_River_basin_bound.geojson", driver='GeoJSON')