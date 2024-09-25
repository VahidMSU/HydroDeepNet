import arcpy
from arcpy.sa import *
import os

### this script is to clip the PRISM data that we generated in PRISM_annual.py to the extent of Michigan state
### input: PRISM data in raster format
### output: clipped raster format


# Define paths
input_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/"
output_path = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI/"

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# List all .tif files in the input directory
rasters = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.tif')]
print("Rasters to clip:", rasters)
# Define the extent of Michigan state
michigan_extent = arcpy.Extent(-90, 40, -81, 49)
print(f"Michigan Extent: {michigan_extent}")

# Set environment settings
arcpy.env.workspace = input_path
arcpy.env.overwriteOutput = True

# Clip each raster
for raster in rasters:
    desc = arcpy.Describe(raster)
    print(f"Processing {raster}")
    print(f"Original Extent: {desc.extent}")

    out_raster = os.path.join(output_path, os.path.basename(raster))
    print(f"Clipping {raster} to {out_raster}")
    arcpy.Clip_management(
        in_raster=raster,
        rectangle=f"{michigan_extent.XMin} {michigan_extent.YMin} {michigan_extent.XMax} {michigan_extent.YMax}",
        out_raster=out_raster,
        in_template_dataset="#",
        nodata_value="#",
        clipping_geometry="NONE",
        maintain_clipping_extent="NO_MAINTAIN_EXTENT"
    )

    clipped_desc = arcpy.Describe(out_raster)
    print(f"Clipped Extent: {clipped_desc.extent}")

print("Done")
