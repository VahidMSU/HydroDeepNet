import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import rasterio
import fiona
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
# Assuming sa.Describe is defined in sa.py

try:
    from SWATGenX.sa import sa
    from SWATGenX.sa import align_rasters
except ImportError:
    from sa import sa
    from sa import align_rasters


def generate_swatplus_rasters(BASE_PATH, VPUID, NAME, LEVEL, MODEL_NAME, landuse_product, landuse_epoch, ls_resolution, dem_resolution):
    print(f"################## Generating raster files for {NAME} {LEVEL} {VPUID} ##################")
    original_landuse_path = os.path.join(BASE_PATH, f"LandUse/{landuse_product}_CONUS/{VPUID}/{landuse_product}_{VPUID}_{landuse_epoch}_{ls_resolution}m.tif")
    original_soil_path = os.path.join(BASE_PATH, f"Soil/gSSURGO_CONUS/{VPUID}/gSSURGO_{VPUID}_{ls_resolution}m.tif")
    original_dem_path = os.path.join(BASE_PATH, f"DEM/VPUID/{VPUID}/")

    # Find the correct DEM file based on the resolution
    dem_names = os.listdir(original_dem_path)
    for dem_name in dem_names:
        if dem_name.endswith(".tif") and f'{dem_resolution}m' in dem_name:
            original_dem_path = os.path.join(original_dem_path, dem_name)
            break

    # Define paths for SWAT+ input
    SOURCE = os.path.join(BASE_PATH, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}")
    swatplus_shapes_path = os.path.join(SOURCE, "Watershed/Shapes/")
    swatplus_landuse_path = os.path.join(SOURCE, "Watershed/Rasters/Landuse/")
    os.makedirs(swatplus_landuse_path, exist_ok=True)
    swatplus_landuse_output = os.path.join(swatplus_landuse_path, "landuse.tif")

    swatplus_dem_input = os.path.join(SOURCE, "Watershed/Rasters/DEM/")
    os.makedirs(swatplus_dem_input, exist_ok=True)
    swatplus_dem_output = os.path.join(swatplus_dem_input, "dem.tif")

    swatplus_soil_input = os.path.join(SOURCE, "Watershed/Rasters/Soil/")
    os.makedirs(swatplus_soil_input, exist_ok=True)
    swatplus_soil_output = os.path.join(swatplus_soil_input, "soil.tif")
    swatplus_soil_temp = os.path.join(swatplus_soil_input, f"soil_{ls_resolution}m_temp.tif")

    swatplus_subbasins_input = os.path.join(swatplus_shapes_path, "SWAT_plus_subbasins.shp")
    watershed_boundary_path = os.path.join(swatplus_shapes_path, "watershed_boundary.shp")

    # Create the watershed boundary
    gdf = gpd.read_file(swatplus_subbasins_input)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    watershed_boundary = box(xmin, ymin, xmax, ymax)
    watershed_boundary = gpd.GeoDataFrame(geometry=[watershed_boundary], crs=gdf.crs)
    watershed_boundary = watershed_boundary.buffer(250)
    watershed_boundary.to_file(watershed_boundary_path)
    spatial_analysis = sa()
    # Set Coordinate System using `sa.Describe`
    desc = spatial_analysis.Describe(watershed_boundary_path)

    # Extract and save DEM using `rasterio` and `mask`
    dem_raster = spatial_analysis.ExtractByMask(original_dem_path, watershed_boundary_path)
    dem_raster.save(swatplus_dem_output)

    # Extract and save Landuse
    landuse_raster = spatial_analysis.ExtractByMask(original_landuse_path, watershed_boundary_path)
    landuse_raster.save(swatplus_landuse_output)

    # Extract and save Soil
    soil_raster = spatial_analysis.ExtractByMask(original_soil_path, watershed_boundary_path)
    soil_raster.save(swatplus_soil_output)

    spatial_analysis.snap_rasters(swatplus_landuse_output, swatplus_soil_output)

    #def resample_dem(path):
    #    assert os.path.exists(path), f"Raster not found: {path}"
    #    ## resample to 25m
    #    os.system(f"gdalwarp -tr 250 250 -r near {path} {path.replace('.tif', '_resampled.tif')}")
    #    print("Done!")
    #    os.remove(path)
    #    os.rename(path.replace('.tif', '_resampled.tif'), path)

    #resample_dem(swatplus_dem_output)

    #sa().snap_rasters(swatplus_dem_output, swatplus_soil_output)

    #align_rasters(swatplus_landuse_output, swatplus_dem_output, swatplus_soil_output)


    # Replace invalid soil cells with the most common value using NumPy and `rasterio`
    with rasterio.open(swatplus_soil_output) as src:
        soil_array = src.read(1)  # Read the first band
        nodata_value = src.nodata if src.nodata is not None else 2147483647
        soil_array[soil_array == nodata_value] = 2147483647

    unique, counts = np.unique(soil_array[soil_array != 2147483647], return_counts=True)
    most_common_value = unique[np.argmax(counts)]
    print(f"Most common value: {most_common_value}")

    # Read Soil CSV for mask values
    path = "/data/SWATGenXApp/GenXAppData/Soil/SWAT_gssurgo.csv"
    df = pd.read_csv(path)
    mask_value = df.muid.values
    print(f"mask value data type: {type(mask_value)}")

    assert most_common_value in mask_value, f"Most common value {most_common_value} is not in SWAT_gssurgo.csv"
    
    ## make it integer
    mask_value = [int(i) for i in mask_value]


    print(f"Number of unique values in SWAT_gssurgo.csv: {len(mask_value)}")
    print(f"Number of unique values in soil raster: {len(np.unique(soil_array))}")
    print(f"Number of unique soil rasters not found in mask:{len(set(np.unique(soil_array)) - set(mask_value))} ")
    assert most_common_value in mask_value, f"Most common value {most_common_value} is not in SWAT_gssurgo.csv"
    print(f"Most common value: {most_common_value}")

    soil_array = np.where(soil_array == 2147483647, most_common_value, soil_array)
    soil_array = np.where(np.isin(soil_array, mask_value), soil_array, most_common_value)
    #soil_array = np.where(soil_array == 2673731, most_common_value, soil_array)


    # Write the updated soil_array back to a raster
    with rasterio.open(swatplus_soil_output) as src:
        transform = src.transform
        crs = src.crs
        profile = src.profile
        profile.update(dtype=rasterio.uint32, compress='lzw')



    with rasterio.open(swatplus_soil_temp, 'w', **profile) as dst:
        dst.write(soil_array, 1)



    # Clip the temp soil raster to watershed boundary and save it using `rasterio.mask.mask`
    with fiona.open(watershed_boundary_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        with rasterio.open(swatplus_soil_temp) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

        with rasterio.open(swatplus_soil_output, "w", **out_meta) as dest:
            dest.write(out_image)

    # Delete the temp soil raster
    if os.path.exists(swatplus_soil_temp):
        os.remove(swatplus_soil_temp)
    print(f"################## Finished generating raster files for {NAME} {LEVEL} {VPUID} ##################")
if __name__ == "__main__":
    VPUID = "0410"
    LEVEL = "huc8"
    NAME = "04100013"
    landuse_product = "NLCD"
    landuse_epoch = "2021"
    ls_resolution = "250"
    dem_resolution = "30"
    MODEL_NAME = "SWAT_MODEL"
    BASE_PATH = r'/data/SWATGenXApp/GenXAppData/'
    generate_swatplus_rasters(BASE_PATH, VPUID, NAME, LEVEL, MODEL_NAME, landuse_product, landuse_epoch, ls_resolution, dem_resolution)
