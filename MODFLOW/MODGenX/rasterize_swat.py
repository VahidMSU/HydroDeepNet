from MODGenX.gdal_operations import gdal_sa as arcpy
import os
import geopandas as gpd
import pandas as pd
import numpy as np
try:
    from MODGenX.utils import *
except ImportError:
    from utils import *

from MODGenX.Logger import Logger

logger = Logger(verbose=True)

def rasterize_SWAT_features(BASE_PATH, feature_type, output_raster_path, load_raster_args):

    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    ref_raster_path = load_raster_args['ref_raster']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']
    NAME = load_raster_args['NAME']
    username = load_raster_args['username']
    VPUID = load_raster_args['VPUID']

    shapefile_paths = generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION, username, VPUID)

    # Define paths based on the type of feature
    if feature_type == "lakes":
        feature_path = shapefile_paths["lakes"]
        logger.info(f'lake: {feature_path}')
        temp_feature_path = os.path.join(f'/data/SWATGenXApp/Users/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
        logger.info(f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
    elif feature_type == "rivers":
        feature_path = shapefile_paths["rivers"]
        logger.info(f'river: {feature_path}')

        temp_feature_path = os.path.join(f'/data/SWATGenXApp/Users/{username}', f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1_modflow.shp')
    else:
        logger.info(f"Unknown feature type: {feature_type}. Supported types are 'lakes' and 'rivers'.")
        return

    modflow_grid_path = shapefile_paths['grids']

    # Set target CRS to EPSG:26990 (NAD83 / Michigan Central)
    target_crs = "EPSG:26990"
    
    # Read the shapefile using geopandas and reproject to target CRS
    try:
        original_feature = gpd.read_file(feature_path)
        logger.info(f"Original feature CRS: {original_feature.crs}")
        
        if original_feature.crs is None or original_feature.crs == "":
            logger.info("Warning: Feature CRS is None or empty. Assuming EPSG:4326 (WGS 84).")
            original_feature.crs = "EPSG:4326"
        
        # Reproject the features to the target CRS
        original_feature = original_feature.to_crs(target_crs)
        logger.info(f"Reprojected feature CRS: {original_feature.crs}")
    except Exception as e:
        logger.info(f"Error reading or reprojecting feature: {e}")
        return

    # Check if original_feature has data
    if len(original_feature) == 0:
        logger.info(f"Warning: No features found in {feature_path}")
        # Create an empty output raster matching the reference raster
        from_reference_raster(ref_raster_path, output_raster_path)
        return

    # Prepare the data for rasterization
    if os.path.exists(modflow_grid_path):
        # Get modflow grid
        modflow_grid = gpd.GeoDataFrame(gpd.read_file(modflow_grid_path))
        # Ensure same CRS
        modflow_grid = modflow_grid.to_crs(target_crs)
        
        try:
            # Perform intersection with modflow grid
            modflow_feature_grid = original_feature.overlay(modflow_grid, how='intersection')
            
            if feature_type == "lakes":
                modflow_feature_grid['lake_area'] = modflow_feature_grid.geometry.area
                modflow_feature_grid['COND'] = modflow_feature_grid['lake_area']
            elif feature_type == "rivers":
                modflow_feature_grid['Len3'] = modflow_feature_grid.geometry.length
                # Ensure the required columns exist
                if 'Wid2' not in modflow_feature_grid.columns or 'Dep2' not in modflow_feature_grid.columns:
                    logger.info(f"Warning: Required columns missing. Using default values.")
                    modflow_feature_grid['COND'] = modflow_feature_grid.geometry.length * 10
                else:
                    modflow_feature_grid['COND'] = (modflow_feature_grid['Wid2'] * modflow_feature_grid['Len3']) / modflow_feature_grid['Dep2']
        except Exception as e:
            logger.info(f"Error performing overlay: {e}")
            modflow_feature_grid = original_feature.copy()
            if feature_type == "lakes":
                modflow_feature_grid['COND'] = modflow_feature_grid.geometry.area
            elif feature_type == "rivers":
                modflow_feature_grid['COND'] = modflow_feature_grid.geometry.length * 10
    else:
        logger.info(f'MODFLOW grid was not created yet. Using SWAT model {feature_type} shapefile for initialization of package.')
        modflow_feature_grid = original_feature.copy()
        
        if feature_type == "lakes":
            modflow_feature_grid['COND'] = modflow_feature_grid.geometry.area
        elif feature_type == "rivers":
            # Check for required columns
            if 'Wid2' in modflow_feature_grid.columns and 'Len2' in modflow_feature_grid.columns and 'Dep2' in modflow_feature_grid.columns:
                modflow_feature_grid['COND'] = (modflow_feature_grid['Wid2'] * modflow_feature_grid['Len2']) / modflow_feature_grid['Dep2']
            else:
                logger.info(f"Warning: Required columns missing. Using geometry length")
                modflow_feature_grid['COND'] = modflow_feature_grid.geometry.length * 10

    # Ensure COND values are positive and reasonable
    modflow_feature_grid['COND'] = modflow_feature_grid['COND'].fillna(10.0)
    modflow_feature_grid['COND'] = modflow_feature_grid['COND'].clip(lower=1.0)
    
    logger.info(f"Feature count for rasterization: {len(modflow_feature_grid)}")
    logger.info(f"COND value range: {modflow_feature_grid['COND'].min()} to {modflow_feature_grid['COND'].max()}")
    
    # Save as a temporary shapefile
    modflow_feature_grid.to_file(temp_feature_path)

    # Configure environment settings using gdal_sa.env
    env = arcpy.env
    env.workspace = BASE_PATH
    env.overwriteOutput = True
    reference_path = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif")
    env.snapRaster = reference_path
    env.cellSize = RESOLUTION
    logger.info(f'#### REFERENCE RESOLUTION FOR RASTERIZATION: {RESOLUTION}')
    env.outputCoordinateSystem = arcpy.Describe(reference_path).spatialReference
    env.nodata = "NONE"
    env.extent = arcpy.Describe(reference_path).extent

    # Convert the temporary shapefile to raster
    try:
        arcpy.PolygonToRaster_conversion(temp_feature_path, "COND", output_raster_path, 
                                         "MAXIMUM_COMBINED_AREA", "NONE", cellsize=RESOLUTION)
        logger.info(f"Successfully rasterized {feature_type} to {output_raster_path}")
    except Exception as e:
        logger.info(f"Error in rasterization: {e}")
        # Create a fallback raster if rasterization fails
        from_reference_raster(reference_path, output_raster_path)

    # Clean up
    try:
        arcpy.Delete_management(temp_feature_path)
    except:
        logger.info(f"Warning: Could not delete temporary file {temp_feature_path}")

def from_reference_raster(ref_raster_path, output_raster_path):
    """Create a zero-filled raster based on a reference raster"""
    from osgeo import gdal
    
    # Open the reference raster
    ref_ds = gdal.Open(ref_raster_path)
    if ref_ds is None:
        logger.info(f"Cannot open reference raster: {ref_raster_path}")
        return
        
    # Get reference raster properties
    driver = gdal.GetDriverByName('GTiff')
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    
    # Create output raster
    out_ds = driver.Create(output_raster_path, cols, rows, 1, gdal.GDT_Float32)
    if out_ds is None:
        logger.info(f"Cannot create output raster: {output_raster_path}")
        return
        
    # Set geotransform and projection
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    
    # Fill raster with zeros
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    out_band.Fill(0)
    
    # Clean up
    out_band = None
    out_ds = None
    ref_ds = None
    
    logger.info(f"Created empty raster at {output_raster_path}")
