import pandas as pd
import geopandas as gpd
huron_river_watershed_path = "/data/MyDataBase/SWATGenXAppData/SWAT_input/huc8/4100013/SWAT_plus_Subbasin/SWAT_plus_Subbasin.shp"
bound = gpd.read_file(huron_river_watershed_path).to_crs('EPSG:26990')
print(f" The size of the watershed is {sum(bound.area)/10**6} km^2")