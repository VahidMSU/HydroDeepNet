import arcpy
import os

RESOLUTION = 250
path = f"{SWATGenXPaths.base_path}Recharge/Recharge_rasterized_{RESOLUTION}m.tif"
reference_raster = f"{SWATGenXPaths.base_path}all_rasters/DEM_{RESOLUTION}m.tif"
output_path = os.path.join("{SWATGenXPaths.base_path}all_rasters", f"Recharge_{RESOLUTION}m.tif")
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = arcpy.Describe(reference_raster).spatialReference
arcpy.env.snapRaster = reference_raster
arcpy.env.extent = arcpy.Describe(reference_raster).extent
arcpy.env.cellSize = RESOLUTION
arcpy.env.nodata = "NONE"

### clip by ExtractByMask
arcpy.CheckOutExtension("Spatial")
arcpy.gp.ExtractByMask_sa(path, reference_raster, output_path)
arcpy.CheckInExtension("Spatial")