from osgeo import gdal
import os

# Paths
soil_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/Soil/soil.tif"
landuse_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/resampled_majority.tif"
output_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/snapped_resampled_majority.tif"

# Step 2: Extract metadata from the reference raster
def get_raster_metadata(raster_path):
    """Extract resolution and extent from a raster."""
    print(f"Extracting metadata from raster: {raster_path}")
    ds = gdal.Open(raster_path)
    if not ds:
        raise RuntimeError(f"Failed to open raster: {raster_path}")
    
    transform = ds.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    xmin, ymax = transform[0], transform[3]
    xmax = xmin + (ds.RasterXSize * xres)
    ymin = ymax + (ds.RasterYSize * yres)
    
    ds = None  # Close dataset
    return xres, yres, xmin, ymin, xmax, ymax


# Step 3: Snap the landuse raster to the soil raster grid
def snap_raster(input_path, output_path, xres, yres, xmin, ymin, xmax, ymax):
    """Snap a raster to a reference grid."""
    print(f"Snapping raster: {input_path} to grid")
    options = gdal.WarpOptions(
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=abs(yres),  # yRes is negative, take absolute
        resampleAlg="near"
    )
    result = gdal.Warp(output_path, input_path, options=options)
    if not result:
        raise RuntimeError(f"Failed to snap raster: {input_path}")
    
    result.FlushCache()
    print(f"Snapped raster saved to: {output_path}")


xres, yres, xmin, ymin, xmax, ymax = get_raster_metadata(soil_path)
snap_raster(landuse_path, output_path, xres, yres, xmin, ymin, xmax, ymax)

## remove initial landuse
#os.remove(landuse_path)
#os.rename(output_path, landuse_path)
print("Done!")
