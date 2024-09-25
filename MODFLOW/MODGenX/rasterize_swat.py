
import arcpy
from arcpy import env
import os
import geopandas as gpd
import pandas as pd
import numpy as np
try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
def rasterize_SWAT_features(BASE_PATH, feature_type, output_raster_path, load_raster_args):

    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    ref_raster_path = load_raster_args['ref_raster']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']
    NAME = load_raster_args['NAME']

    shapefile_paths = generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION)

    # Define paths based on the type of feature

    if feature_type == "lakes":
        feature_path = shapefile_paths ["lakes"]
        print('lake:',feature_path)
        temp_feature_path =  os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
        print(f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
        #print(arcpy.Describe(temp_feature_path).shapeType)

    elif feature_type == "rivers":
        feature_path = shapefile_paths ["rivers"]
        print('river:',feature_path)
        temp_feature_path = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1_modflow.shp')
    else:
        print(f"Unknown feature type: {feature_type}. Supported types are 'lakes' and 'rivers'.")
        return

    modflow_grid_path = shapefile_paths ['grids']

    # Set overwrite to true for arcpy
    # Read the shapefile using geopandas

    original_feature = gpd.read_file(feature_path)

    if os.path.exists(modflow_grid_path):

        modflow_grid = gpd.GeoDataFrame(pd.read_pickle(modflow_grid_path))
        modflow_feature_grid = original_feature.overlay(modflow_grid, how='intersection')

        if feature_type == "lakes":
            modflow_feature_grid['lake_area'] = modflow_feature_grid.geometry.area
            modflow_feature_grid['COND'] = modflow_feature_grid['lake_area']
        elif feature_type == "rivers":
            modflow_feature_grid['Len3'] = modflow_feature_grid.geometry.length
            modflow_feature_grid['COND'] = (modflow_feature_grid['Wid2'] * modflow_feature_grid['Len3']) / modflow_feature_grid['Dep2']

    else:

        print(f'MODFLOW grid was not created yet. Using SWAT model {feature_type} shapefile for initialization of package.')
        modflow_feature_grid = original_feature.copy()
        if feature_type == "lakes":
            modflow_feature_grid['lake_full_area'] = modflow_feature_grid.geometry.area
            modflow_feature_grid['COND'] = modflow_feature_grid['lake_full_area']
        elif feature_type == "rivers":
            modflow_feature_grid['COND'] = (modflow_feature_grid['Wid2'] * modflow_feature_grid['Len2']) / modflow_feature_grid['Dep2']
    # Save as a temporary shapefile
    modflow_feature_grid.to_file(temp_feature_path)



    # Use the cell size of the reference raster
    #cell_size = arcpy.GetRasterProperties_management(ref_raster_path, "CELLSIZEX").getOutput(0)

    env.workspace = BASE_PATH
    arcpy.env.overwriteOutput = True  # Enable overwrite
    reference_path = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif")
    arcpy.env.snapRaster = reference_path  # Clear the snap raster setting
    arcpy.env.cellSize = RESOLUTION  # Setting cell size
    print('#### REFERENCE RESOLUTION FOR RASTERIZATION:',RESOLUTION)
    arcpy.env.outputCoordinateSystem = arcpy.Describe(reference_path).spatialReference
    arcpy.env.nodata = "NONE"
    arcpy.env.overwriteOutput = True
    arcpy.env.extent = arcpy.Describe(reference_path).extent


    # Convert the temporary shapefile to raster
    if feature_type == "lakes":
        arcpy.PolygonToRaster_conversion(temp_feature_path, "COND", output_raster_path, cellsize=RESOLUTION)
    elif feature_type == "rivers":
        arcpy.PolylineToRaster_conversion(temp_feature_path, "COND", output_raster_path, cellsize=RESOLUTION)

    # Delete the temporary shapefile
    arcpy.Delete_management(temp_feature_path)
