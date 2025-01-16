import os 
import osgeo
path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0415/huc12/04292000/SWAT_MODEL/Watershed/Rasters/DEM/dem.tif"
def resample_dem(path):
    assert os.path.exists(path), f"Raster not found: {path}"
    ## resample to 25m
    os.system(f"gdalwarp -tr 25 25 -r near {path} {path.replace('.tif', '_resampled.tif')}")
    print("Done!")
    os.remove(path)
    os.rename(path.replace('.tif', '_resampled.tif'), path)