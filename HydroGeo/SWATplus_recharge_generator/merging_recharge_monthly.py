import arcpy
import os

def create_zero_raster(reference_raster, resolution):
    # Create a zero raster based on the reference raster
    zero_raster = os.path.join(output_dir, f"zero_raster_{resolution}.tif")
    zero_raster_sa = arcpy.sa.CreateConstantRaster(-1, "FLOAT", reference_raster)
    zero_raster_sa.save(zero_raster)

    return zero_raster
def create_mosaic(rasters_to_add, output_dir, ver_num, year, mon, resolution, reference_raster):
    arcpy.env.snapRaster = reference_raster
    arcpy.env.cellSize = reference_raster
    arcpy.env.overwriteOutput = True
    arcpy.env.extent = reference_raster
    arcpy.env.nodata = -1
    output_path = os.path.join(output_dir, f"recharge_{year}_{mon}_{ver_num}_{resolution}m.tif")
    arcpy.MosaicToNewRaster_management(rasters_to_add, output_dir, os.path.basename(output_path),
                                    pixel_type="32_BIT_FLOAT", number_of_bands=1, mosaic_method="MEAN")


# Assuming ver_num is always 0, adjust if necessary
def merge_rasters(ver_num, year, mon,resolution, NAMES, VPUID, reference_raster):

    list_of_unfound_directories = []
    for NAME in NAMES:
        if NAME in list_of_unfound_directories:
            continue
        if not os.path.exists(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/recharg_output_SWAT_gwflow_MODEL"):
            print(f"Directory not found for {NAME}")
            list_of_unfound_directories.append(NAME)
            continue
        verifications = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/recharg_output_SWAT_gwflow_MODEL/verification_stage_{ver_num}"
        if not os.path.exists(verifications):
            print(f"Verification directory not found for {NAME}")
            continue
        # Read all files ending with tif using arcpy
        arcpy.env.workspace = verifications
        if rasters := arcpy.ListRasters(f"recharge_{year}_{mon}_{resolution}m.tif"):
            rasters_to_add.extend([os.path.join(verifications, raster) for raster in rasters])


    if len(rasters_to_add) > 1:  # Ensures we have rasters to add
        create_mosaic(rasters_to_add, output_dir,ver_num, year, mon, resolution, reference_raster)


import itertools
if __name__ == "__main__":
    path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
    NAMES = os.listdir(path)
    NAMES.remove('log.txt')
    VPUID = "0000"
    resolution = 30
    # Ensure the output directory exists
    output_dir = "/data/MyDataBase/SWATGenXAppData//all_rasters"
    reference_raster = f"/data/MyDataBase/SWATGenXAppData//all_rasters/DEM_{resolution}m.tif"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zero_raster = create_zero_raster(reference_raster, resolution)

    list_of_unfound_directories = []
    for ver_num in [0]:
        for mon, year in itertools.product(range(1, 13), range(2004, 2021)):
            rasters_to_add = [zero_raster]
            merge_rasters(ver_num, year, mon, resolution, NAMES, VPUID, reference_raster)
            print(f"Merging completed for year {year}.")
            print(f"List of unfound directories: {list_of_unfound_directories}")
            break

    print("All merging completed.")