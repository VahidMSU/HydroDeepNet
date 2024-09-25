import geopandas as gpd	
path = "/data/MyDataBase/SWATGenXAppData/Grid/Centroid_DEM_250m.shp"
gdf = gpd.read_file(path)
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y
gdf.to_file(path)