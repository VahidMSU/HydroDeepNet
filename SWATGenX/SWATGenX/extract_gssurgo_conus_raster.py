from osgeo import gdal
import os
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths

def extract_CONUS_gssurgo_raster():
    """
    Extract the gSSURGO raster from the original geodatabase to a GeoTIFF file.
    Required GDAL compiled with FileGDB support.
    Uses memory-efficient processing for large rasters.
    """
    gSSURGO_raster = SWATGenXPaths.gSSURGO_raster

    # Input and output paths
    gSSURGO_CONUS_gdb_path = SWATGenXPaths.gSSURGO_CONUS_gdb_path
    assert os.path.exists(gSSURGO_CONUS_gdb_path), f"gSSURGO_CONUS_gdb_path does not exist: {gSSURGO_CONUS_gdb_path}"
    input_raster = f"OpenFileGDB:{gSSURGO_CONUS_gdb_path}:MURASTER_30m"

    # Set creation options for optimized processing
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            'TILED=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'COMPRESS=LZW',
            'BIGTIFF=YES'
        ],
        callback=gdal.TermProgress_nocb
    )

    # Process the raster with the optimized settings
    gdal.Translate(gSSURGO_raster, input_raster, options=translate_options)
    print(f"Raster exported to {gSSURGO_raster}")

if __name__ == "__main__":
    extract_CONUS_gssurgo_raster()