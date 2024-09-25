import arcpy
import os
from arcpy.sa import ExtractByMask

RESOLUTION = 250
reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"

path_of_rasters = '/data/MyDataBase/SWATGenXAppData/NHDPlusData/Michigan/'

name_of_rasters = os.listdir(path_of_rasters)
name_of_rasters = [name for name in name_of_rasters if name.endswith(".tif")]
arcpy.env.overwriteOutput = True
arcpy.env.workspace = path_of_rasters
arcpy.env.extent = reference_raster
arcpy.env.snapRaster = reference_raster
arcpy.env.cellSize = reference_raster

# get the cell size of the reference raster
cellSize = arcpy.GetRasterProperties_management(reference_raster, "CELLSIZEX")

for name in name_of_rasters:
	print(f"Processing {name}")
	raster_path = f"{path_of_rasters}{name}"
	temp = f"/data/MyDataBase/SWATGenXAppData/all_rasters/{name.split('.')[0]}_MILP_{RESOLUTION}m_temp.tif"
	output_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/{name.split('.')[0]}_MILP_{RESOLUTION}m.tif"
	# first resample to reference raster reslution
	arcpy.Resample_management(raster_path, temp,cellSize, "NEAREST")
	arcpy.env.extent = reference_raster
	ExtractByMask(temp, reference_raster).save(output_raster)
	arcpy.Delete_management(temp)
