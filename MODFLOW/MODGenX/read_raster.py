#!/usr/bin/env python3
"""
Utility script for examining raster files to diagnose issues.
This script checks for multiple bands, inspects data ranges, 
and can help identify problems with DEM files.
"""

import os
import sys
import numpy as np
import rasterio
import argparse
from osgeo import gdal

def examine_raster(raster_path, verbose=False):
    """
    Examine a raster file and report its properties
    """
    print(f"\nExamining raster: {raster_path}")
    
    if not os.path.exists(raster_path):
        print(f"ERROR: File does not exist: {raster_path}")
        return
    
    try:
        # Open with rasterio for detailed info
        with rasterio.open(raster_path) as src:
            print(f"Driver: {src.driver}")
            print(f"Dimensions: {src.width} x {src.height} pixels")
            print(f"Bands: {src.count}")
            print(f"Coordinate system: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"NoData value: {src.nodata}")
            
            # Check each band
            for i in range(1, src.count + 1):
                band = src.read(i)
                band_min, band_max = band.min(), band.max()
                band_mean = band.mean()
                unique_values = len(np.unique(band))
                
                print(f"\nBand {i}:")
                print(f"  Data type: {band.dtype}")
                print(f"  Min: {band_min}, Max: {band_max}, Mean: {band_mean:.2f}")
                print(f"  Unique values: {unique_values}")
                
                # Warning for suspiciously high values in elevation data
                if band_max > 10000 and i == 1 and "DEM" in raster_path:
                    print(f"  WARNING: Very high elevation values detected ({band_max})")
                
                # Check if this might be an alpha band (constant values)
                if unique_values == 1 and band_min == 255:
                    print(f"  WARNING: Band {i} has constant value of 255 - likely an alpha channel")
                
                # Print histogram
                if verbose:
                    hist, bins = np.histogram(band, bins=10)
                    print(f"  Histogram: {hist}")
                    print(f"  Bins: {bins}")
    
    except Exception as e:
        print(f"Error examining raster with rasterio: {str(e)}")
    
    try:
        # Also use GDAL for additional checks
        ds = gdal.Open(raster_path)
        if ds is not None:
            print("\nGDAL Information:")
            print(f"GDAL driver: {ds.GetDriver().ShortName}")
            metadata = ds.GetMetadata()
            print(f"Metadata: {metadata}")
            
            # Check color interpretation - can help identify alpha bands
            for i in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(i)
                color_interp = band.GetColorInterpretation()
                print(f"Band {i} color interpretation: {gdal.GetColorInterpretationName(color_interp)}")
            
            ds = None
    except Exception as e:
        print(f"Error examining raster with GDAL: {str(e)}")

def fix_multi_band_dem(input_raster, output_raster, band_number=1):
    """
    Create a single band DEM from a multi-band raster
    """
    print(f"Creating single-band raster from band {band_number} of {input_raster}")
    
    try:
        # Use gdal_translate to extract just one band
        gdal.Translate(output_raster, input_raster, bandList=[band_number])
        print(f"Created single-band raster: {output_raster}")
        
        # Verify the output
        examine_raster(output_raster)
        
    except Exception as e:
        print(f"Error creating single-band raster: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Examine and fix raster files")
    parser.add_argument("raster_path", help="Path to the raster file to examine")
    parser.add_argument("--fix", "-f", help="Create a new single-band raster from the first band", action="store_true")
    parser.add_argument("--output", "-o", help="Output path for the fixed raster")
    parser.add_argument("--band", "-b", help="Band number to extract (default: 1)", type=int, default=1)
    parser.add_argument("--verbose", "-v", help="Show more detailed information", action="store_true")
    
    args = parser.parse_args()
    
    examine_raster(args.raster_path, args.verbose)
    
    if args.fix:
        output_path = args.output if args.output else os.path.splitext(args.raster_path)[0] + "_fixed.tif"
        fix_multi_band_dem(args.raster_path, output_path, args.band)

if __name__ == "__main__":
    main()
