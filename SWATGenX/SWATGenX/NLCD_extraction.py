import arcpy
import os
import glob
import geopandas as gpd
import pyproj
import numpy as np
from shapely.geometry import box

class NLCDExtractor:
    def __init__(self, VPUID, epoch):
        self.VPUID = VPUID
        self.epoch = epoch
        self.NLCD_all_files_path = (
            "/data/SWATGenXApp/GenXAppData/LandUse/NLCD_landcover_2021_release_all_files_20230630/"
        )
        self.HUC4_base_path = fr"/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
        self.temp_path = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{VPUID}/rect.shp"
        self.original_out_raster = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{VPUID}/NLCD_{VPUID}_{epoch}.tif"
        self.VPUID_DEM_base = f"/data/SWATGenXApp/GenXAppData/DEM/VPUID/{VPUID}"
        self.NLCD_path = None
        self.spatial_ref = None
        self.HUC4_path = None
        self.HUC4 = None
        self.HUC4_crs = None

        arcpy.env.overwriteOutput = True
        ### use DEM spatial reference to the environment
        DEM_path = self.get_DEM_path()
        DEM_crs = self.get_DEM_crs(DEM_path)
        arcpy.env.outputCoordinateSystem = DEM_crs

    def get_DEM_path(self):
        DEM_name = os.listdir(self.VPUID_DEM_base)
        for file in DEM_name:
            if file.endswith(".tif") and "500m" in file:
                DEM_path = os.path.join(self.VPUID_DEM_base, file)
                break
        return DEM_path

    def get_DEM_crs(self, DEM_path):
        return arcpy.Describe(DEM_path).spatialReference

    def NLCD_extract_by_VPUID(self):
        NLCD_all_files = os.listdir(self.NLCD_all_files_path)
        for file in NLCD_all_files:
            if file.endswith(".img") and str(self.epoch) == file.split("_")[1]:
                self.NLCD_name = file
                self.NLCD_path = os.path.join(self.NLCD_all_files_path, self.NLCD_name)
                break

        self.spatial_ref = arcpy.Describe(self.NLCD_path).spatialReference
        print(self.spatial_ref.exporttostring())  ## output: Albers_Conical_Equal_Area

        HUC4_name = os.listdir(self.HUC4_base_path)
        for name in HUC4_name:
            if name.endswith(".gdb"):
                self.HUC4_path = os.path.join(self.HUC4_base_path, name)
                break
        print(self.HUC4_path)
        self.HUC4 = gpd.read_file(self.HUC4_path, layer='WBDHU4')
        self.HUC4_crs = self.HUC4.crs
        print(self.HUC4_crs)  ## output: EPSG:4269

        self.HUC4 = self.HUC4.to_crs(self.spatial_ref.exporttostring())
        print(self.HUC4.crs)  ## output: Albers_Conical_Equal_Area

        ## now create a rectangular object for clipping the raster
        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)])
        rect.crs = self.spatial_ref.exporttostring()

        os.makedirs(os.path.dirname(self.temp_path), exist_ok=True)
        rect.to_file(self.temp_path)

        ### now use the rect to clip the raster
        ## now clip the raster
        arcpy.Clip_management(self.NLCD_path, self.temp_path, self.original_out_raster, "#", "#", "NONE")

        DEM_path = self.get_DEM_path()
        target_crs = self.get_DEM_crs(DEM_path)
        ### now project the original_out_raster to the target_crs and save it
        output_path = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{self.VPUID}/NLCD_{self.VPUID}_{self.epoch}_30m.tif"
        arcpy.ProjectRaster_management(self.original_out_raster, output_path, target_crs)

        ### now resample the raster to 100m, 200m, 500m, 1000m, 2000m
        resolutions = [100, 250, 500, 1000, 2000]
        for resolution in resolutions:
            output_path = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{self.VPUID}/NLCD_{self.VPUID}_{self.epoch}_{resolution}m.tif"
            if not os.path.exists(output_path):
                arcpy.Resample_management(self.original_out_raster, output_path, resolution, "NEAREST")

# Usage
def NLCD_extract_by_VPUID(VPUID, epoch):
    original_out_raster = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{VPUID}/NLCD_{VPUID}_{epoch}.tif"
    if not os.path.exists(original_out_raster):
        extractor = NLCDExtractor(VPUID, epoch)
        extractor.NLCD_extract_by_VPUID()
    else:
        print(f"NLCD data for {VPUID} and {epoch} already exists.")
if __name__ == "__main__":
    VPUID = "0407"
    epoch = 2021
    NLCD_extract_by_VPUID(VPUID, epoch)