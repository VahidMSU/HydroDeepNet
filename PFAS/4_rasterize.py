import arcpy
import os
import pandas as pd
import multiprocessing

path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/compounds/"

# Function to create a raster from a shapefile
def create_raster(shapefile):
	in_features = os.path.join(path, 'shapes', shapefile)
	out_raster = os.path.join(path, "rasters", f"{shapefile[:-4]}.tif")
	arcpy.env.overwriteOutput = True
	cellsize = 0.00027778
	arcpy.FeatureToRaster_conversion(in_features, "val", out_raster, cellsize)
 

if __name__ == '__main__':
	# Get a list of shapefiles in the path
	shapefiles = [file for file in os.listdir(os.path.join(path, 'shapes')) if file.endswith(".shp")]

	# Create a pool of worker processes
	pool = multiprocessing.Pool(processes=50)

	# Map the create_raster function to each shapefile in parallel
	pool.map(create_raster, shapefiles)

	# Close the pool of worker processes
	pool.close()
	pool.join()
