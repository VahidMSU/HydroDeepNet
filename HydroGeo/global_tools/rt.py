

import arcpy
def extract_raster(raster_path, out_raster, RESOLUTION, overwrite=False):
	""" this function extract the raster data based on the reference raster
	Args:
		raster_path (str): the path to the raster file
		out_raster (str): the path to save the extracted raster
		RESOLUTION (int): the resolution of the raster
	Returns:
		None
	None: The input raster must be larger to have the same row and columns as the reference raster
	"""
	referebce_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"
	arcpy.env.snapRaster = referebce_raster
	arcpy.env.cellSize = referebce_raster
	arcpy.env.overwriteOutput = overwrite
	arcpy.env.extent = referebce_raster
 
	arcpy.env.outputCoordinateSystem = referebce_raster
	arcpy.sa.ExtractByMask(raster_path, referebce_raster).save(out_raster)
	print(f"Raster {raster_path} extracted successfully")
	
