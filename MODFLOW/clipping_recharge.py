import GDAL
import os

RESOLUTION = 250
path = f"/data/SWATGenXApp/GenXAppData/Recharge/Recharge_rasterized_{RESOLUTION}m.tif"
reference_raster = f"/data/SWATGenXApp/GenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"
output_path = os.path.join("/data/SWATGenXApp/GenXAppData/all_rasters", f"Recharge_{RESOLUTION}m.tif")
GDAL.env.overwriteOutput = True
GDAL.env.outputCoordinateSystem = GDAL.Describe(reference_raster).spatialReference
GDAL.env.snapRaster = reference_raster
GDAL.env.extent = GDAL.Describe(reference_raster).extent
GDAL.env.cellSize = RESOLUTION
GDAL.env.nodata = "NONE"

### clip by ExtractByMask
GDAL.CheckOutExtension("Spatial")
GDAL.gp.ExtractByMask_sa(path, reference_raster, output_path)
GDAL.CheckInExtension("Spatial")