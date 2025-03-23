"""
Helper functions for working with raster data in MODGenX.
These functions make it easier to handle raster processing operations.
"""

import numpy as np
import os
import rasterio
from osgeo import gdal
from MODGenX.logger_singleton import get_logger

logger = get_logger()

def examine_raster(raster_path, get_stats=True, plot_histogram=False):
    """
    Examine a raster file and report its properties.
    
    Parameters:
    -----------
    raster_path : str
        Path to the raster file
    get_stats : bool, optional
        Whether to calculate statistics for each band
    plot_histogram : bool, optional
        Whether to create a histogram plot
        
    Returns:
    --------
    dict
        Dictionary containing raster properties
    """
    result = {
        'path': raster_path,
        'exists': os.path.exists(raster_path),
        'valid': False,
        'bands': [],
        'error': None
    }
    
    if not result['exists']:
        result['error'] = f"File does not exist: {raster_path}"
        return result
    
    try:
        with rasterio.open(raster_path) as src:
            result['driver'] = src.driver
            result['width'] = src.width
            result['height'] = src.height
            result['count'] = src.count
            result['crs'] = str(src.crs)
            result['transform'] = src.transform
            result['nodata'] = src.nodata
            
            # Get band information
            for i in range(1, src.count + 1):
                band = src.read(i)
                band_info = {
                    'band_num': i,
                    'dtype': str(band.dtype)
                }
                
                if get_stats:
                    band_info['min'] = float(band.min())
                    band_info['max'] = float(band.max())
                    band_info['mean'] = float(band.mean())
                    band_info['unique_values'] = int(len(np.unique(band)))
                
                result['bands'].append(band_info)
            
            # Plot histogram if requested
            if plot_histogram:
                import matplotlib.pyplot as plt
                for i in range(1, src.count + 1):
                    band = src.read(i)
                    hist, bins = np.histogram(band[band != src.nodata], bins=20)
                    result['bands'][i-1]['histogram'] = {
                        'counts': hist.tolist(),
                        'bins': bins.tolist()
                    }
        
        result['valid'] = True
        return result
    
    except Exception as e:
        result['error'] = str(e)
        return result

def extract_band(input_raster, output_raster, band_number=1):
    """
    Extract a single band from a multi-band raster.
    
    Parameters:
    -----------
    input_raster : str
        Path to the input raster file
    output_raster : str
        Path to save the output raster file
    band_number : int, optional
        Band number to extract (1-based)
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        gdal.Translate(output_raster, input_raster, bandList=[band_number])
        logger.info(f"Extracted band {band_number} from {input_raster} to {output_raster}")
        return True
    except Exception as e:
        logger.error(f"Error extracting band {band_number}: {str(e)}")
        return False

def match_raster_extent(base_raster_path, target_raster_path, output_raster_path, 
                        resampling=gdal.GRA_NearestNeighbour):
    """
    Match the extent and resolution of a target raster to a base raster.
    
    Parameters:
    -----------
    base_raster_path : str
        Path to the base raster file
    target_raster_path : str
        Path to the target raster file
    output_raster_path : str
        Path to save the output raster file
    resampling : gdal.ResampleAlg, optional
        Resampling algorithm to use
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Get base raster info
        base_ds = gdal.Open(base_raster_path)
        if base_ds is None:
            logger.error(f"Could not open base raster: {base_raster_path}")
            return False
            
        base_geo = base_ds.GetGeoTransform()
        base_proj = base_ds.GetProjection()
        base_xsize = base_ds.RasterXSize
        base_ysize = base_ds.RasterYSize
        
        # Warp options to match base raster
        warp_options = gdal.WarpOptions(
            width=base_xsize,
            height=base_ysize,
            outputBounds=(base_geo[0], base_geo[3] + base_ysize * base_geo[5], 
                         base_geo[0] + base_xsize * base_geo[1], base_geo[3]),
            dstSRS=base_proj,
            resampleAlg=resampling
        )
        
        # Perform warping
        gdal.Warp(output_raster_path, target_raster_path, options=warp_options)
        logger.info(f"Matched raster extent from {target_raster_path} to {output_raster_path}")
        
        base_ds = None
        return True
        
    except Exception as e:
        logger.error(f"Error matching raster extent: {str(e)}")
        return False
