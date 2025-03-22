from MODGenX.gdal_operations import gdal_sa as GDAL
import os
import geopandas as gpd
import numpy as np
from osgeo import gdal, osr, ogr
import shutil
try:
    from MODGenX.utils import generate_shapefile_paths
except ImportError:
    from utils import generate_shapefile_paths
from MODGenX.Logger import Logger

logger = Logger(verbose=True)

def rasterize_SWAT_features(BASE_PATH, feature_type, output_raster_path, load_raster_args):
    """
    Rasterize SWAT features (lakes or rivers) to match reference raster dimensions exactly.
    
    Parameters:
    -----------
    BASE_PATH : str
        Base path for data
    feature_type : str
        Type of feature to rasterize ('lakes' or 'rivers')
    output_raster_path : str
        Path where output raster will be saved
    load_raster_args : dict
        Dictionary containing various arguments needed for processing
    """
    # Extract parameters from load_raster_args
    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    ref_raster_path = load_raster_args['ref_raster']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']
    NAME = load_raster_args['NAME']
    username = load_raster_args['username']
    VPUID = load_raster_args['VPUID']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    
    # Generate paths for shapefiles based on feature type
    shapefile_paths = generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION, username, VPUID)
    
    # Validate reference raster exists
    if not os.path.exists(ref_raster_path):
        logger.info(f"Reference raster not found: {ref_raster_path}")
        return
    
    # Get the appropriate feature path based on feature_type
    if feature_type == "lakes":
        feature_path = shapefile_paths["lakes"]
        temp_feature_path = os.path.join(f'/data/SWATGenXApp/Users/{username}', 
                                         f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/lakes1_modflow.shp')
        logger.info(f'lake: {feature_path}')
    elif feature_type == "rivers":
        feature_path = shapefile_paths["rivers"]
        temp_feature_path = os.path.join(f'/data/SWATGenXApp/Users/{username}', 
                                         f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1_modflow.shp')
        logger.info(f'river: {feature_path}')
    else:
        logger.info(f"Unknown feature type: {feature_type}. Supported types are 'lakes' and 'rivers'.")
        return
    
    # STEP 1: Analyze reference raster to get exact dimensions and spatial properties
    logger.info(f"Analyzing reference raster: {ref_raster_path}")
    ref_ds = gdal.Open(ref_raster_path)
    if ref_ds is None:
        logger.info(f"Cannot open reference raster: {ref_raster_path}")
        return
    
    ref_rows = ref_ds.RasterYSize
    ref_cols = ref_ds.RasterXSize
    ref_geotransform = ref_ds.GetGeoTransform()
    ref_projection = ref_ds.GetProjection()
    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromWkt(ref_projection)
    
    # Get spatial properties for coordinate transformations
    x_min = ref_geotransform[0]
    y_max = ref_geotransform[3]
    x_res = ref_geotransform[1]
    y_res = ref_geotransform[5]
    x_max = x_min + (ref_cols * x_res)
    y_min = y_max + (ref_rows * y_res)
    
    logger.info(f"Reference raster properties - Size: {ref_rows}x{ref_cols}, Resolution: {x_res}x{y_res}")
    logger.info(f"Reference raster bounds: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    # STEP 2: Check if feature file exists and has data
    if not os.path.exists(feature_path):
        logger.info(f"Feature shapefile not found: {feature_path}")
        create_empty_raster(ref_ds, output_raster_path)
        ref_ds = None
        return
    
    try:
        # Read shapefile with geopandas
        gdf = gpd.read_file(feature_path)
        
        if len(gdf) == 0:
            logger.info(f"No features found in {feature_path}")
            create_empty_raster(ref_ds, output_raster_path)
            ref_ds = None
            return
        
        # Ensure CRS is set
        if gdf.crs is None:
            logger.info(f"CRS not defined in shapefile. Assuming EPSG:4326")
            gdf.crs = "EPSG:4326"
        
        # Get target CRS from reference raster
        if ref_srs.IsProjected():
            target_crs = f"EPSG:{ref_srs.GetAuthorityCode('PROJCS')}" if ref_srs.GetAuthorityCode('PROJCS') else "EPSG:26990"
        else:
            target_crs = f"EPSG:{ref_srs.GetAuthorityCode('GEOGCS')}" if ref_srs.GetAuthorityCode('GEOGCS') else "EPSG:4326"
        
        # Reproject to match reference raster projection
        logger.info(f"Reprojecting features from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
        
        # Add COND field for rasterization
        if feature_type == "lakes":
            gdf['COND'] = gdf.geometry.area
        else:  # rivers
            if 'Wid2' in gdf.columns and 'Len2' in gdf.columns and 'Dep2' in gdf.columns:
                gdf['COND'] = (gdf['Wid2'] * gdf['Len2']) / gdf['Dep2']
            else:
                logger.info("Required columns missing. Using geometry length.")
                gdf['COND'] = gdf.geometry.length * 10
        
        # Ensure COND values are valid
        gdf['COND'] = gdf['COND'].fillna(10.0)
        gdf['COND'] = gdf['COND'].clip(lower=1.0)
        
        # Save temporary shapefile
        logger.info(f"Saving temporary shapefile with {len(gdf)} features")
        gdf.to_file(temp_feature_path)
        
    except Exception as e:
        logger.info(f"Error processing feature shapefile: {str(e)}")
        create_empty_raster(ref_ds, output_raster_path)
        ref_ds = None
        return
    
    # STEP 3: Create a blank raster with exact reference dimensions
    logger.info("Creating base raster with reference dimensions")
    driver = gdal.GetDriverByName('GTiff')
    temp_raster = output_raster_path + ".temp.tif"
    
    out_ds = driver.Create(temp_raster, ref_cols, ref_rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_geotransform)
    out_ds.SetProjection(ref_projection)
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    band = None
    out_ds = None
    
    # STEP 4: Burn the vector features into the raster
    logger.info(f"Rasterizing features to match reference dimensions")
    try:
        # Open vector dataset
        vector_ds = ogr.Open(temp_feature_path)
        layer = vector_ds.GetLayer()
        
        # Open output raster for burning
        raster_ds = gdal.Open(temp_raster, gdal.GA_Update)
        
        # Burn vector features into raster
        err = gdal.RasterizeLayer(
            raster_ds,                  # Output raster dataset
            [1],                        # List of bands to burn values into
            layer,                      # Input layer
            options=["ATTRIBUTE=COND"]  # Use COND attribute for pixel values
        )
        
        if err != 0:
            logger.info(f"Error rasterizing layer: {err}")
        
        # Close datasets
        raster_ds = None
        vector_ds = None
        
        # Validate output raster has correct dimensions
        check_ds = gdal.Open(temp_raster)
        if check_ds:
            out_rows = check_ds.RasterYSize
            out_cols = check_ds.RasterXSize
            logger.info(f"Output raster dimensions: {out_rows}x{out_cols}")
            
            if out_rows != ref_rows or out_cols != ref_cols:
                logger.info(f"WARNING: Output dimensions {out_rows}x{out_cols} don't match reference {ref_rows}x{ref_cols}")
            check_ds = None
        
        # Move temp raster to final location
        shutil.move(temp_raster, output_raster_path)
        logger.info(f"Successfully created raster at {output_raster_path}")
        
    except Exception as e:
        logger.info(f"Error during rasterization: {str(e)}")
        # Fall back to empty raster if rasterization fails
        create_empty_raster(ref_ds, output_raster_path)
    
    # Clean up
    try:
        if os.path.exists(temp_feature_path):
            GDAL.Delete_management(temp_feature_path)
        if os.path.exists(temp_raster):
            GDAL.Delete_management(temp_raster)
    except Exception as e:
        logger.info(f"Warning: Could not delete temporary files: {str(e)}")
    
    ref_ds = None

def create_empty_raster(ref_ds, output_path):
    """
    Create an empty raster with the same dimensions and spatial reference as the reference raster.
    
    Parameters:
    -----------
    ref_ds : gdal.Dataset
        Reference dataset to copy properties from
    output_path : str
        Path where the empty raster will be saved
    """
    logger.info(f"Creating empty raster at {output_path}")
    
    if ref_ds is None:
        logger.info("Reference dataset is None, cannot create empty raster")
        return
    
    # Get reference properties
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # Create new raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    
    if out_ds is None:
        logger.info(f"Could not create output raster at {output_path}")
        return
    
    # Set spatial properties
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Fill with zeros
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)
    
    # Clean up
    band = None
    out_ds = None
    
    logger.info(f"Created empty raster with dimensions {rows}x{cols}")
