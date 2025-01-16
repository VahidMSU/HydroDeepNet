from osgeo import gdal
import os

# Paths
base_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/"
original_DEM_path = os.path.join(base_path, "DEM/dem.tif")
reference_raster = os.path.join(base_path, "Soil/soil.tif")
resampled_raster_majority = os.path.join(base_path, "DEM/resampled_majority.tif")
resampled_raster_nearest = os.path.join(base_path, "DEM/resampled_nearest.tif")

# Step 1: Extract metadata from the reference raster
def get_raster_metadata(raster_path):
    """Extract resolution, extent, and transform from a raster."""
    ds = gdal.Open(raster_path)
    if not ds:
        raise RuntimeError(f"Failed to open raster: {raster_path}")

    transform = ds.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    xmin, ymax = transform[0], transform[3]
    xmax = xmin + (ds.RasterXSize * xres)
    ymin = ymax + (ds.RasterYSize * yres)

    ds = None  # Close dataset
    return xres, yres, xmin, ymin, xmax, ymax, transform

# Get metadata for the reference raster
xres, yres, xmin, ymin, xmax, ymax, transform = get_raster_metadata(reference_raster)

# Adjust the bounds to align with the reference raster's pixel grid
def align_bounds(xmin, ymin, xmax, ymax, xres, yres):
    """Align bounds to match the grid of the reference raster."""
    xmin_aligned = xmin - (xmin % xres)
    ymin_aligned = ymin - (ymin % abs(yres))
    xmax_aligned = xmax - (xmax % xres)
    ymax_aligned = ymax - (ymax % abs(yres))
    return xmin_aligned, ymin_aligned, xmax_aligned, ymax_aligned

# Align bounds
xmin, ymin, xmax, ymax = align_bounds(xmin, ymin, xmax, ymax, xres, yres)

# Step 2: Resample to reference raster resolution and extent (Majority method) with snapping
def resample_raster(input_path, output_path, xres, yres, xmin, ymin, xmax, ymax, resample_alg):
    """Resample a raster to a specific resolution and extent, snapping to grid."""
    options = gdal.WarpOptions(
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=abs(yres),
        resampleAlg=resample_alg,
        targetAlignedPixels=True  # Ensures snapping to reference raster grid
    )
    result = gdal.Warp(output_path, input_path, options=options)
    if not result:
        raise RuntimeError(f"Failed to resample raster: {input_path}")
    result.FlushCache()
    print(f"Resampled raster saved to: {output_path}")



# Resample using the majority method
print("Resampling to reference resolution and extent with majority method...")
resample_raster(
    input_path=original_DEM_path,
    output_path=resampled_raster_majority,
    xres=xres,
    yres=yres,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    resample_alg="mode"  # Majority method in GDAL
)








# Paths
soil_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/Soil/soil.tif"
majority_DEM = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/resampled_majority.tif"
snapped_output_path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/snapped_resampled_majority.tif"

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
snap_raster(majority_DEM, snapped_output_path, xres, yres, xmin, ymin, xmax, ymax)

## remove initial landuse
os.remove(majority_DEM)
os.rename(snapped_output_path, majority_DEM)
print("Done!")


# Step 3: Resample back to original resolution (Nearest method)
original_xres, original_yres, *_= get_raster_metadata(original_DEM_path)

print("Resampling back to original resolution with nearest method...")
resample_raster(
    input_path=resampled_raster_majority,
    output_path=resampled_raster_nearest,
    xres=original_xres,
    yres=original_yres,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    resample_alg="near"  # Nearest neighbor method
)

# Step 4: Cleanup and finalize
print("Cleaning up and finalizing...")
# Uncomment the following lines to remove temporary files and finalize
os.remove(original_DEM_path)  # Remove the initial DEM
os.remove(resampled_raster_majority)  # Remove majority resampled DEM
os.rename(resampled_raster_nearest, original_DEM_path)  # Rename nearest resampled DEM to original DEM

print("All operations completed successfully!")