import arcpy
import os
from arcpy.sa import *
RESOLUTION = 250
YEARS = [1990, 2000, 2010]

# Set source, reference, and output paths
for YEAR in YEARS:
	source_path = f"/data/MyDataBase/SWATGenXAppData/US_population/pden{YEAR}_block/pden{YEAR}_block/"
	file_name = os.listdir(source_path)
	### only end with tif
	file_name = [f for f in file_name if f.endswith(".tif")]
	source_path = os.path.join(source_path, file_name[0])
	reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"
	output_path = "/data/MyDataBase/SWATGenXAppData/US_population/extracted"

	# Create output directory if it doesn't exist
	os.makedirs(output_path, exist_ok=True)

	# Set the arcpy environment settings
	arcpy.env.workspace = output_path
	arcpy.env.overwriteOutput = True
	arcpy.env.snapRaster = reference_raster
	arcpy.env.cellSize = reference_raster
	arcpy.env.extent = reference_raster
	arcpy.env.outputCoordinateSystem = arcpy.Describe(reference_raster).spatialReference

	# Extract by mask and save the result
	extracted_raster = ExtractByMask(source_path, reference_raster)
	extracted_raster.save(os.path.join(output_path, f"pden{YEAR}_ML_{RESOLUTION}m.tif"))

	# Check in the Spatial Analyst extension
	arcpy.CheckInExtension("Spatial")
