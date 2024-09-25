import arcpy
import os
# this script is to project the PRISM data to Michigan State Plane Coordinate System 26990 
arcpy.env.workspace = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI/"	
rasters = arcpy.ListRasters()
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(26990)
arcpy.env.overwriteOutput = True
projection = arcpy.SpatialReference(26990)
os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI_projected", exist_ok=True)
for raster in rasters:
	out_raster = os.path.join("/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI_projected", raster)
	arcpy.ProjectRaster_management(raster, out_raster, projection)
	print(f"Projected {raster} to {out_raster}")
	