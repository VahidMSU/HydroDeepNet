import os
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr
try:
    from MODGenX.utils import *
except ImportError:
    from utils import *

def rasterize_SWAT_features(BASE_PATH, feature_type, output_raster_path, load_raster_args, config=None):
    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    ref_raster_path = load_raster_args['ref_raster']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']
    NAME = load_raster_args['NAME']

    shapefile_paths = generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION, config)

    # Define paths based on the type of feature
    if feature_type == "lakes":
        feature_path = shapefile_paths["lakes"]
        print('lake:', feature_path)
        if config is not None:
            temp_feature_path = config.construct_path("SWAT_input", LEVEL, NAME, SWAT_MODEL_NAME, "Watershed/Shapes/lakes1_modflow.shp")
        else:
            temp_feature_path = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
        print(f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
    elif feature_type == "rivers":
        feature_path = shapefile_paths["rivers"]
        print('river:', feature_path)
        if config is not None:
            temp_feature_path = config.construct_path("SWAT_input", LEVEL, NAME, SWAT_MODEL_NAME, "Watershed/Shapes/rivs1_modflow.shp")
        else:
            temp_feature_path = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1_modflow.shp')
    else:
        print(f"Unknown feature type: {feature_type}. Supported types are 'lakes' and 'rivers'.")
        return

    modflow_grid_path = shapefile_paths['grids']

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

    # Get reference raster properties
    reference_path = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif")
    ref_ds = gdal.Open(reference_path)
    ref_transform = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_xsize = ref_ds.RasterXSize
    ref_ysize = ref_ds.RasterYSize
    
    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster_path, ref_xsize, ref_ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_transform)
    out_ds.SetProjection(ref_proj)
    
    # Set nodata value and initialize
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    band.Fill(np.nan)
    
    # Open the temporary shapefile
    vector_ds = ogr.Open(temp_feature_path)
    layer = vector_ds.GetLayer()
    
    # Rasterize
    gdal.RasterizeLayer(out_ds, [1], layer, options=["ATTRIBUTE=COND"])
    
    # Close datasets
    out_ds = None
    vector_ds = None
    ref_ds = None
    
    # Delete the temporary shapefile
    if os.path.exists(temp_feature_path):
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            temp_file = temp_feature_path.replace('.shp', ext)
            if os.path.exists(temp_file):
                os.remove(temp_file)
