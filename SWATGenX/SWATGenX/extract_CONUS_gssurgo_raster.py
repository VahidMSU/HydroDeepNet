from osgeo import gdal

def extract_CONUS_gssurgo_raster(input_raster, output_raster):
    """
        Extract the gSSURGO raster from the original geodatabase to a GeoTIFF file.
        Required GDAL compiled with FileGDB support.
        
    """
    # Input and output paths
    input_raster = "OpenFileGDB:/data/SWATGenXApp/GenXAppData/Soil/gSSURGO_CONUS/gSSURGO_CONUS.gdb:MapunitRaster_30m"

    output_raster = "/data/SWATGenXApp/GenXAppData/Soil/gSSURGO_CONUS/MapunitRaster_30m.tif"
    if not output_raster.endswith(".tif"):
        gdal.Translate(output_raster, input_raster)
        print(f"Raster exported to {output_raster}")
