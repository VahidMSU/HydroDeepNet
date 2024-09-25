
import pandas as pd
import os
import geopandas as gpd	
path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/"
for stat in ['Min', 'Max', 'Mean', 'Std']:
	gdf_min = pd.read_pickle(os.path.join(path, f'PFAS_{stat}.pkl'))
	gdf_min = gpd.GeoDataFrame(gdf_min, crs='EPSG:4326', geometry=gpd.points_from_xy(gdf_min.geometry.x, gdf_min.geometry.y))
	# Define the raster size
	unique_compounds = gdf_min['compound'].unique()
	### create a shapefile for each compound (with the name of the compound as the shapename.shp) and the value of the concentration as the val in the shapefile
	for compound in unique_compounds:
		# Create a GeoDataFrame for the compound
		compound_gdf = gpd.GeoDataFrame(gdf_min[gdf_min['compound'] == compound], crs=gdf_min.crs)
		compound_gdf.rename(columns={'concentration': 'val'}, inplace=True)
		### convert to 4326
		compound_gdf.to_file(os.path.join(path,"compounds",'shapes' ,f"{stat}_{compound}.shp"))
		# Get the bounds of the compound GeoDataFrame
print("Done")