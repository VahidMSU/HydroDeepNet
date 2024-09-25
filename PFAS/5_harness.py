import arcpy
from arcpy.sa import *
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import multiprocessing

def process_raster(in_raster, tem_out_raster, reference_raster, workspace, overwrite_output, extent, snap_raster, cell_size, mask, output_coordinate_system, out_raster):
	# Set the environment settings
	arcpy.env.workspace = workspace
	arcpy.env.overwriteOutput = overwrite_output
	arcpy.env.extent = extent
	arcpy.env.snapRaster = snap_raster
	arcpy.env.cellSize = cell_size
	arcpy.env.mask = mask
	arcpy.env.outputCoordinateSystem = output_coordinate_system
	
	# Project the raster to the reference raster
	arcpy.ProjectRaster_management(in_raster, tem_out_raster, reference_raster)
	ExtractByMask(tem_out_raster, reference_raster).save(out_raster)
	# Delete the temporary raster
	arcpy.Delete_management(tem_out_raster)
	print(f"Raster {raster} has been masked")

if __name__ == '__main__':
	path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/compounds/rasters"
	out_path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/compounds/masked_rasters"
	os.makedirs(out_path, exist_ok=True)
	
	## Read the raster files
	rasters = [file for file in os.listdir(path) if file.endswith(".tif")]  # example Min_CAS376067_PFTeA.tif
	RESOLUTION = [50, 100]
	
	# Create a multiprocessing pool with the number of available CPU cores
	pool = multiprocessing.Pool(processes=50)
	
	for res in RESOLUTION:
		reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/upscaled/DEM_{res}m.tif"
		print(rasters)
		
		for raster in rasters:
			in_raster = os.path.join(path, raster)
			tem_out_raster = os.path.join(out_path, f"_{raster}")
			out_raster = os.path.join(out_path, f"{res}_{raster}")

			# Use the multiprocessing pool to process each raster in parallel
			pool.apply_async(process_raster, (in_raster, tem_out_raster, reference_raster, 
											   "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/compounds/",
											   True, reference_raster, reference_raster, reference_raster,
											   reference_raster, reference_raster, out_raster))	

	# Close the multiprocessing pool and wait for all processes to finish
	pool.close()
	pool.join()

	print("Done")
