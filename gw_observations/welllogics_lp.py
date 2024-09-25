import arcpy
import pandas as pd
import geopandas as gpd
import os
import arcpy
import pandas as pd
import geopandas as gpd
import os
def merge_wellogic_data(well_path):
	shapes = os.listdir(well_path)
	pathes = [shape for shape in shapes if shape.endswith('.shp')]
	all_wells = pd.DataFrame()
	for geodata in pathes:
		print(geodata)
		gdf = gpd.read_file(os.path.join(well_path, geodata)).to_crs('EPSG:4326')
		print(gdf.columns)
		all_wells = pd.concat([all_wells, gdf], axis=0)

	all_wells.to_file(os.path.join(well_path, 'all_wells.shp'))
	all_wells.to_pickle(os.path.join(well_path, 'all_wells.pkl'))

	return all_wells.head()
def load_geodataframe(temp, all_wells_path, columns):
	df = pd.read_pickle(all_wells_path)
	df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
	df['lat'] = df.geometry.x
	df['lon'] = df.geometry.y
	df = df.to_crs('EPSG:26990')
	df['x'] = df.geometry.x
	df['y'] = df.geometry.y
	## create a raster with value as x and another with value as y
	df[columns].to_file(temp)

if __name__ == "__main__":
	all_wells_path = "/data/MyDataBase/SWATGenXAppData/well_info/all_wells.pkl"
	cellSize = 250
	well_path = r"/data/MyDataBase/SWATGenXAppData/WellLogic"
	temp = r"/data/MyDataBase/SWATGenXAppData/WellLogic/x_y/x_y.shp"
	columns=['lat', 'lon', 'x', 'y', 'geometry']
	#df = load_geodataframe(temp, all_wells_path, columns)
	workspace = r"/data/MyDataBase/SWATGenXAppData/WellLogic/x_y"
	os.makedirs(workspace, exist_ok=True)
	reference_raster = r'/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif'
	arcpy.env.workspace = workspace
	arcpy.env.overwriteOutput = True
	arcpy.env.cellSize = cellSize
	arcpy.env.extent = reference_raster
	arcpy.env.snapRaster = reference_raster
	## create a raster with x
	all_rasters_path = r"/data/MyDataBase/SWATGenXAppData/all_rasters"
	arcpy.PointToRaster_conversion(in_features=temp, value_field='lat', out_rasterdataset= os.path.join(all_rasters_path, f'obs_lat_{cellSize}m.tif'), cellsize=250)
	arcpy.PointToRaster_conversion(in_features=temp, value_field='lon', out_rasterdataset= os.path.join(all_rasters_path, f'obs_lon_{cellSize}m.tif'), cellsize=250)
	arcpy.PointToRaster_conversion(in_features=temp, value_field='x', 	out_rasterdataset= os.path.join(all_rasters_path, f'obs_x_{cellSize}m.tif'), cellsize=250)
	arcpy.PointToRaster_conversion(in_features=temp, value_field='y', 	out_rasterdataset= os.path.join(all_rasters_path, f'obs_y_{cellSize}m.tif'), cellsize=250)
