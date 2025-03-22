from MODGenX.Logger import Logger
from MODGenX.gdal_operations import gdal_sa as arcpy
import os
import numpy as np
from osgeo import gdal, osr

logger = Logger(verbose=True)

def clip_raster_by_another(BASE_PATH, raster_path, in_masking, output_path):
    """
    This function creates a new raster by masking an existing one.
    It extracts by extent and ensures the same number of rows and columns as in_masking.
    Also ensures the output is a single-band raster by extracting only the first band.
    """
    env = arcpy.env  # Use gdal_sa's env class
    env.overwriteOutput = True
    os.makedirs(os.path.join("_temp/"), exist_ok=True)
    current_directory = BASE_PATH
    env.workspace = current_directory
    
    # Open the input and mask rasters
    mask_ds = gdal.Open(in_masking)
    if mask_ds is None:
        logger.error(f"Cannot open reference raster: {in_masking}")
        raise ValueError(f"Cannot open reference raster: {in_masking}")
    
    input_ds = gdal.Open(raster_path)
    if input_ds is None:
        logger.error(f"Cannot open input raster {raster_path}")
        raise ValueError(f"Cannot open input raster {raster_path}")
    
    # Get the CRS information
    mask_srs = osr.SpatialReference(wkt=mask_ds.GetProjection())
    input_srs = osr.SpatialReference(wkt=input_ds.GetProjection())
    
    # Set consistent axis mapping
    mask_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    input_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    logger.info(f"Reference raster CRS: {mask_srs.ExportToProj4()}")
    logger.info(f"Input raster CRS: {input_srs.ExportToProj4()}")
    
    # Check if CRS match
    crs_match = input_srs.IsSame(mask_srs)
    logger.info(f"CRS match: {1 if crs_match else 0}")
    
    # Get number of bands
    input_band_count = input_ds.RasterCount
    logger.info(f"Input raster has {input_band_count} bands")
    
    # Get mask raster's geotransform and dimensions
    mask_gt = mask_ds.GetGeoTransform()
    mask_width = mask_ds.RasterXSize
    mask_height = mask_ds.RasterYSize
    
    mask_x_min = mask_gt[0]
    mask_y_max = mask_gt[3]
    mask_x_max = mask_x_min + mask_gt[1] * mask_width
    mask_y_min = mask_y_max + mask_gt[5] * mask_height
    
    # Get input raster's nodata value
    input_band = input_ds.GetRasterBand(1)
    input_nodata = input_band.GetNoDataValue()
    if input_nodata is None:
        input_nodata = -9999  # Default nodata value if none specified
        logger.info(f"No nodata value found in input, using {input_nodata}")
    
    # Prepare temporary file path
    temp_file = os.path.join("_temp", f"temp_{os.path.basename(output_path)}")
    
    # Reproject and resample the input raster to match the mask raster's CRS and resolution
    if not crs_match:
        logger.info("Reprojecting input raster to match reference CRS")
        warp_options = gdal.WarpOptions(
            dstSRS=mask_ds.GetProjection(),
            xRes=mask_gt[1],
            yRes=abs(mask_gt[5]),
            outputBounds=(mask_x_min, mask_y_min, mask_x_max, mask_y_max),
            dstNodata=input_nodata,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        gdal.Warp(temp_file, input_ds, options=warp_options)
    else:
        # Just resample to match resolution and extent
        warp_options = gdal.WarpOptions(
            xRes=mask_gt[1],
            yRes=abs(mask_gt[5]),
            outputBounds=(mask_x_min, mask_y_min, mask_x_max, mask_y_max),
            dstNodata=input_nodata,
            resampleAlg=gdal.GRA_NearestNeighbour
        )
        gdal.Warp(temp_file, input_ds, options=warp_options)
    
    # Validate the temporary output
    temp_ds = gdal.Open(temp_file)
    if temp_ds is None:
        logger.error(f"Failed to create temporary raster: {temp_file}")
        raise ValueError(f"Failed to create temporary raster: {temp_file}")
    
    # Extract the first band if multi-band
    if input_band_count > 1:
        logger.info(f"Extracting first band from {input_band_count}-band raster")
        gdal.Translate(output_path, temp_file, bandList=[1])
    else:
        # Just copy the temporary file to output
        gdal.Translate(output_path, temp_file)
    
    # Verify the output
    out_ds = gdal.Open(output_path)
    if out_ds is None:
        logger.error(f"Failed to create output raster: {output_path}")
        raise ValueError(f"Failed to create output raster: {output_path}")
    
    # Check the output statistics
    out_band = out_ds.GetRasterBand(1)
    out_nodata = out_band.GetNoDataValue()
    
    # Read data for validation
    out_data = out_band.ReadAsArray()
    valid_mask = (out_data != out_nodata) if out_nodata is not None else np.ones(out_data.shape, dtype=bool)
    valid_data = out_data[valid_mask]
    
    if valid_data.size > 0:
        min_val = np.nanmin(valid_data)
        max_val = np.nanmax(valid_data)
        mean_val = np.nanmean(valid_data)
        std_val = np.nanstd(valid_data)
        logger.info(f"Output raster stats: min={min_val}, max={max_val}, mean={mean_val}, stddev={std_val}")
    else:
        logger.warning("No valid data found in output raster!")
    
    # Clean up
    input_ds = None
    mask_ds = None
    temp_ds = None
    out_ds = None
    
    # Remove temporary file
    try:
        os.remove(temp_file)
    except:
        logger.info(f"Could not delete temporary file {temp_file}")
    
    # Make a final check for the output file
    if not os.path.exists(output_path):
        logger.error(f"Output file does not exist: {output_path}")
        raise ValueError(f"Output file does not exist: {output_path}")
    
    logger.info(f"Successfully created clipped raster: {output_path}")
    return output_path
