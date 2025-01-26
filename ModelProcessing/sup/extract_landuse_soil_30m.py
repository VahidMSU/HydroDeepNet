import os
from osgeo import gdal

gdal.UseExceptions()

# Function to enforce alignment to DEM
def enforce_alignment(input_raster, output_raster, reference_dem):
    dem_ds = gdal.Open(reference_dem)
    gt = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    x_res = gt[1]
    y_res = abs(gt[5])
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + x_res * dem_ds.RasterXSize
    ymin = ymax - y_res * dem_ds.RasterYSize
    dem_ds = None

    gdal.Warp(
        output_raster,
        input_raster,
        xRes=x_res,
        yRes=y_res,
        outputBounds=[xmin, ymin, xmax, ymax],
        targetAlignedPixels=True,
        dstSRS=projection,
        resampleAlg="near",  # Nearest neighbor for categorical data
    )
    print(f"Aligned raster: {output_raster} to {reference_dem}")


# Iterate through each directory
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12")
NAMES.remove("log.txt")  # Remove unwanted files

for NAME in NAMES:


    # Define file paths
    soil_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/Soil/soil.tif"
    landuse_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/Landuse/landuse.tif"
    reference_raster = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/DEM/dem.tif"
    resample_raster_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/DEM/dem_30m.tif"
    projected_raster_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/DEM/dem_30m_projected.tif"
    reference_soil = f"/data2/MyDataBase/SWATGenXAppData/all_rasters/gSURRGO_swat_30m.tif"
    reference_landuse = f"/data2/MyDataBase/SWATGenXAppData/all_rasters/landuse_30m.tif"
    aligned_soil = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/Soil/soil_30m.tif"
    aligned_landuse = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/Landuse/landuse_30m.tif"

    # Ensure files exist
    for path in [soil_path, landuse_path, reference_raster, reference_soil, reference_landuse]:
        assert os.path.exists(path), f"File not found: {path}"

    try:
        # Resample the DEM
        gdal.Warp(resample_raster_path, reference_raster, xRes=30, yRes=30, resampleAlg="near")
        print(f"Resampled DEM for {NAME}")

        # Enforce alignment of soil and landuse to DEM grid
        enforce_alignment(reference_soil, aligned_soil, resample_raster_path)
        enforce_alignment(reference_landuse, aligned_landuse, resample_raster_path)

        print(f"Processed {NAME}")

    except Exception as e:
        print(f"Error processing {NAME}: {e}")
