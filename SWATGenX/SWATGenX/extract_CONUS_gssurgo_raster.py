from osgeo import gdal
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
def extract_CONUS_gssurgo_raster():
    """
        Extract the gSSURGO raster from the original geodatabase to a GeoTIFF file.
        Required GDAL compiled with FileGDB support.
        
    """
    gSSURGO_raster = SWATGenXPaths.gSSURGO_raster

    if not gSSURGO_raster.endswith(".tif"):
        # Input and output paths
        input_raster = "OpenFileGDB:/data/SWATGenXApp/GenXAppData/Soil/gSSURGO_CONUS/gSSURGO_CONUS.gdb:MapunitRaster_30m"

        gdal.Translate(gSSURGO_raster, input_raster)
        print(f"Raster exported to {gSSURGO_raster}")
