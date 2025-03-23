from MODGenX.logger_singleton import get_logger
from MODGenX.gdal_operations import gdal_sa as arcpy
import os
import numpy as np
from osgeo import gdal, osr

# Use the singleton logger pattern instead of directly initializing Logger
logger = get_logger()

def clip_raster_by_another(BASE_PATH, raster_path, in_masking, output_path):
    """
    This function creates a new raster by masking an existing one.
    It extracts by extent and ensures the same number of rows and columns as in_masking.
    Also ensures the output is a single-band raster by extracting only the first band.
    """
    logger.warning(f"Clipping raster {raster_path} by {in_masking}")
    logger.info(f"Output path: {output_path}")
    
    env = arcpy.env  # Use gdal_sa's env class
    env.overwriteOutput = True
    os.makedirs(os.path.join("_temp/"), exist_ok=True)
    current_directory = BASE_PATH
    env.workspace = current_directory
    
    # Open the input and mask rasters
    mask_ds = gdal.Open(in_masking)
    assert mask_ds is not None, f"Cannot open mask raster {in_masking}"

    input_ds = gdal.Open(raster_path)
    assert input_ds is not None, f"Cannot open input raster {raster_path}"
    
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
    
    # Check the output statistics without using ReadAsArray
    out_band = out_ds.GetRasterBand(1)
    
    # Use GetStatistics which doesn't require gdal_array
    try:
        # Force computation of statistics (approx=False, force=True)
        stats = out_band.GetStatistics(False, True)
        min_val, max_val, mean_val, std_val = stats
        logger.info(f"Output raster stats: min={min_val}, max={max_val}, mean={mean_val}, stddev={std_val}")
        
        if min_val == max_val:
            logger.warning("Output raster has constant values, which might indicate a problem")
    except Exception as e:
        logger.warning(f"Could not compute statistics: {str(e)}")
        
        # Alternate method: scan a sample of pixels to check for valid data
        width = out_band.XSize
        height = out_band.YSize
        
        # Sample pixels to check validity
        sample_size = min(100, width * height)
        has_valid_data = False
        
        for _ in range(sample_size):
            x = int(np.random.uniform(0, width))
            y = int(np.random.uniform(0, height))
            # Read a single pixel value
            data = out_band.ReadRaster(x, y, 1, 1, buf_type=gdal.GDT_Float32)
            if data:
                import struct
                value = struct.unpack('f', data)[0]
                if not (np.isnan(value) or np.isinf(value)):
                    has_valid_data = True
                    break
        
        if not has_valid_data:
            logger.warning("No valid data found in sampled pixels!")
    
    # Check the data range in the output
    min_max = out_band.ComputeRasterMinMax()
    if "domain.tif" not in output_path: ### only for this raster, we excpet everything be 1
        logger.info(f"Clipped raster data range: {min_max[0]} to {min_max[1]}")
        assert min_max[0] != min_max[1], "Output raster has constant values"
        
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

