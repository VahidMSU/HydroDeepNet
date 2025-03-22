#!/usr/bin/env python3
"""
Utility script to fix the DEM_250m.tif file by ensuring it's a single band raster
with valid elevation values.
"""

import os
import sys
import numpy as np
from osgeo import gdal
import rasterio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_dem(dem_path, output_path=None):
    """
    Fix the DEM raster by ensuring it's a single band with valid values
    """
    if output_path is None:
        base_dir = os.path.dirname(dem_path)
        basename = os.path.basename(dem_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(base_dir, f"{name}_fixed{ext}")
    
    # Create backup
    backup_path = dem_path + ".backup"
    if not os.path.exists(backup_path):
        try:
            import shutil
            shutil.copy2(dem_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Could not create backup: {str(e)}")
    
    # Open the DEM file
    try:
        src_ds = gdal.Open(dem_path)
        if src_ds is None:
            logger.error(f"Could not open {dem_path}")
            return False
        
        # Check how many bands
        band_count = src_ds.RasterCount
        logger.info(f"DEM has {band_count} bands")
        
        # Check the first band
        band = src_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        data_min, data_max = data.min(), data.max()
        logger.info(f"Band 1 data range: {data_min} to {data_max}")
        
        # Extract only the first band and save to output
        logger.info(f"Extracting band 1 to {output_path}")
        gdal.Translate(output_path, src_ds, bandList=[1])
        
        # Check the output
        with rasterio.open(output_path) as fixed_src:
            fixed_data = fixed_src.read(1)
            fixed_min, fixed_max = fixed_data.min(), fixed_data.max()
            logger.info(f"Fixed DEM data range: {fixed_min} to {fixed_max}")
            
            # Verify it looks reasonable
            if fixed_min < -900000 or fixed_max > 900000:
                logger.error(f"Fixed DEM still has extreme values! Manual intervention needed.")
                return False
            
            logger.info(f"DEM fixed successfully: {output_path}")
            return True
    except Exception as e:
        logger.error(f"Error fixing DEM: {str(e)}")
        return False

def inspect_dem_values(dem_path):
    """
    Detailed inspection of DEM values
    """
    try:
        with rasterio.open(dem_path) as src:
            logger.info(f"DEM details:")
            logger.info(f"- Dimensions: {src.width} x {src.height}")
            logger.info(f"- CRS: {src.crs}")
            logger.info(f"- Transform: {src.transform}")
            logger.info(f"- Bands: {src.count}")
            
            for i in range(1, src.count + 1):
                band = src.read(i)
                logger.info(f"Band {i}:")
                logger.info(f"- Data type: {band.dtype}")
                logger.info(f"- Min: {band.min()}, Max: {band.max()}, Mean: {np.mean(band)}")
                logger.info(f"- NoData value: {src.nodata}")
                
                # Count values in reasonable ranges
                reasonable = np.sum((band > 0) & (band < 5000))
                extreme = np.sum((band < -10000) | (band > 10000))
                logger.info(f"- Values in reasonable range (0-5000): {reasonable} ({reasonable/band.size*100:.2f}%)")
                logger.info(f"- Values in extreme range: {extreme} ({extreme/band.size*100:.2f}%)")
                
                # Check for alpha band (all 255)
                if band.min() == 255 and band.max() == 255:
                    logger.info(f"- WARNING: Band {i} appears to be an alpha channel (all values = 255)")
            
            # Check for specific problem values
            problem_values = np.unique(np.where((band < -10000) | (band > 10000), band, 0))
            if len(problem_values) > 0:
                logger.info(f"Problem values: {problem_values}")
    except Exception as e:
        logger.error(f"Error inspecting DEM: {str(e)}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: fix_dem.py DEM_PATH [OUTPUT_PATH]")
        return 1
    
    dem_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    logger.info(f"Inspecting DEM: {dem_path}")
    inspect_dem_values(dem_path)
    
    if fix_dem(dem_path, output_path):
        logger.info("DEM fixed successfully")
        return 0
    else:
        logger.error("Failed to fix DEM")
        return 1

if __name__ == "__main__":
    sys.exit(main())
