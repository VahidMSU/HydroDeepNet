from osgeo import gdal
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
def extract_CONUS_gssurgo_raster():
    """
        Extract the gSSURGO raster from the original geodatabase to a GeoTIFF file.
        Required GDAL compiled with FileGDB support.
        
    """
    gSSURGO_raster = SWATGenXPaths.gSSURGO_raster

    if not gSSURGO_raster.endswith(".tif"):
        # Input and output paths
        gSSURGO_CONUS_gdb_path = SWATGenXPaths.gSSURGO_CONUS_gdb_path 
        input_raster = f"OpenFileGDB:{gSSURGO_CONUS_gdb_path}:MapunitRaster_30m"

        gdal.Translate(gSSURGO_raster, input_raster)
        print(f"Raster exported to {gSSURGO_raster}")
