path = "/home/rafieiva/MyDataBase/codes/gw_machine_learning/PFAS_GNN/Biosolid leaching/Biosolids_Sites_HRU.shp"
import pandas as pd
import geopandas as gpd
gdf = gpd.read_file(path)
gdf = gdf.to_crs("EPSG:4326")   # convert to WGS84
## 
gdf ['geometry'] = gdf['geometry'].to_crs("EPSG:26990").centroid.to_crs("EPSG:4326")

gdf['Industry'] = "Biosolid"

gdf = gdf[['geometry', 'Industry']]
## save to geojson
output__name = "/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites.geojson"
gdf.to_file(output__name, driver='GeoJSON')