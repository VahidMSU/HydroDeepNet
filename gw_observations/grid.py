import arcpy
import os

# Define paths and workspace settings
RESOLUTION = 250
reference_raster = fr'/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif'
workspace = r"/data/MyDataBase/SWATGenXAppData/all_rasters"

# Set ArcPy environment settings
arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True
arcpy.env.cellSize = reference_raster
arcpy.env.extent = reference_raster
arcpy.env.snapRaster = reference_raster

# Convert raster to point features
temp_point_features = f"/data/MyDataBase/SWATGenXAppData/Grid/Centroid_DEM_{RESOLUTION}m.shp"

# Convert point features back to raster for x and y coordinate rasters
arcpy.PointToRaster_conversion(
    in_features=temp_point_features, 
    value_field='x', 
    out_rasterdataset=os.path.join(workspace, f'x_{RESOLUTION}m.tif'), 
    cellsize=RESOLUTION
)
arcpy.PointToRaster_conversion(
    in_features=temp_point_features, 
    value_field='y', 
    out_rasterdataset=os.path.join(workspace, f'y_{RESOLUTION}m.tif'), 
    cellsize=RESOLUTION
)

# Cleanup: Delete the temporary shapefile
arcpy.Delete_management(temp_point_features)
