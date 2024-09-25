import arcpy
import os

# Set the workspace containing the rasters
arcpy.env.workspace = "/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI_resampled_30m/"
rasters = arcpy.ListRasters()

RESOLUTION = 50

# Path to the reference raster with the desired resolution and extent
reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/upscaled/DEM_{RESOLUTION}m.tif"

# Define the output path
output_path = f"/data/MyDataBase/SWATGenXAppData/codes/PRISM/annual/total_ppt_MI_resampled_{RESOLUTION}m/"
os.makedirs(output_path, exist_ok=True)

# Set environment settings to match the reference raster
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(26990)
arcpy.env.overwriteOutput = True
arcpy.env.extent = arcpy.Describe(reference_raster).extent
arcpy.env.cellSize = arcpy.Describe(reference_raster).meanCellWidth
arcpy.env.snapRaster = reference_raster

# Iterate through each raster in the workspace
for raster in rasters:
    print(os.path.join(output_path, raster.split(".")[0] + f"_{RESOLUTION}m.tif"))
    # Copy the raster with the environment settings applied
    out_raster = os.path.join(output_path, raster.split(".")[0].split('_30m')[0] + f"_{RESOLUTION}m.tif")
    arcpy.management.CopyRaster(
        in_raster=raster,
        
        out_rasterdataset=	out_raster,
        config_keyword="",
        background_value="",
        nodata_value="-9999",
        onebit_to_eightbit="NONE",
        colormap_to_RGB="NONE",
        pixel_type="32_BIT_FLOAT",
        scale_pixel_value="NONE",
        RGB_to_Colormap="NONE",
        format="TIFF",
        transform=""
    )
    
    print(f"Copied {raster} to {out_raster}")

print("Done")
