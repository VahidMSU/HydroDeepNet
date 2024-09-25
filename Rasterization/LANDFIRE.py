import arcpy
from arcpy.sa import *	
import os

root_path = "/data/MyDataBase/SWATGenXAppData/LANDFIRE_database/"
RESOLUTIONS = [30, 250]
extracted_rasters_path = os.path.join(root_path, "extracted_rasters")

# Ensure the output directory exists
os.makedirs(extracted_rasters_path, exist_ok=True)

# Get the directories inside the root path
directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

# Search inside each directory for files ending with .tif
for directory in directories:
    print(f"Processing {directory}")
    directory_path = os.path.join(root_path, directory)
    files = [f for f in os.listdir(directory_path) if f.endswith(".tif")]
    
    for file in files:
        print(f"Processing {file}")
        file_path = os.path.join(directory_path, file)
        print(file_path)
        
        for RESOLUTION in RESOLUTIONS:
            output_raster = os.path.join(extracted_rasters_path, f"{file[:-4]}_{RESOLUTION}m.tif")
            reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{RESOLUTION}m.tif"
            
            arcpy.env.workspace = extracted_rasters_path
            arcpy.env.overwriteOutput = True
            arcpy.env.snapRaster = reference_raster
            arcpy.env.cellSize = reference_raster

            # Set extent to match the reference raster
            desc = arcpy.Describe(reference_raster)
            arcpy.env.extent = desc.extent
            arcpy.env.outputCoordinateSystem = desc.spatialReference
            
            # Extract by mask
            arcpy.CheckOutExtension("Spatial")
            outExtractByMask = ExtractByMask(file_path, reference_raster)
            outExtractByMask.save(output_raster)
            arcpy.CheckInExtension("Spatial")

print("Processing complete.")
