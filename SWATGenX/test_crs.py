import os


dem_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Rasters/DEM/dem.tif"

landuse_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Rasters/Landuse/landuse.tif"

soil_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Rasters/Soil/soil.tif"

### check the crs of the rasters
import rasterio

def check_crs(raster_path):
    with rasterio.open(raster_path) as src:
        crs = src.crs
    return crs

print("DEM CRS:")
print(check_crs(dem_path))
print("Landuse CRS:")
print(check_crs(landuse_path))
print("Soil CRS:")
print(check_crs(soil_path))



### now shapes

stream_path =    "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Shapes/SWAT_plus_streams.shp"
subbasin_path =  "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Shapes/SWAT_plus_subbasins.shp"
watershed_path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID//0405/huc12/04097970/SWAT_MODEL_Web_Application/Watershed/Shapes/SWAT_plus_watersheds.shp"

assert os.path.exists(stream_path), "Stream path does not exist"
assert os.path.exists(subbasin_path), "Subbasin path does not exist"
assert os.path.exists(watershed_path), "Watershed path does not exist"


import geopandas as gpd

stream_gdf = gpd.read_file(stream_path)
subbasin_gdf = gpd.read_file(subbasin_path)
watershed_gdf = gpd.read_file(watershed_path)

print("Stream CRS:")
print(stream_gdf.crs)
print("Subbasin CRS:")
print(subbasin_gdf.crs)
print("Watershed CRS:")
print(watershed_gdf.crs)