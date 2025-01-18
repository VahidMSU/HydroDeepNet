import os
import glob
import geopandas as gpd
import pyproj
import numpy as np
from shapely.geometry import box
import os
import glob
import geopandas as gpd
import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import box

def check_NLCD_by_VPUID(VPUID, epoch):
    basedirectory = "/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/"
    #"Y:\SWATGenXApp\GenXAppData\LandUse\NLCD_CONUS\0206\NLCD_0206_2021_500m.tif"
    RESOLUTIONS = [30, 100, 250, 500, 1000, 2000]
    for resolution in RESOLUTIONS:
        if not os.path.exists(f"{basedirectory}{VPUID}/NLCD_{VPUID}_{epoch}_{resolution}m.tif"):
            raise ValueError(f"{basedirectory}{VPUID}/NLCD_{VPUID}_{epoch}_{resolution}m.tif does not exist")
    print(f"#### NLCD data exists for {VPUID} #####")

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

    def get_DEM_path(self):
        DEM_name = os.listdir(self.VPUID_DEM_base)
        for file in DEM_name:
            if file.endswith(".tif") and "500m" in file:
                DEM_path = os.path.join(self.VPUID_DEM_base, file)
                break
        return DEM_path

    def get_DEM_crs(self, DEM_path):
        raster = gdal.Open(DEM_path)
        return raster.GetProjection()
    
    def get_EPSG(self):
        import pandas as pd 
        base_input_raster = f'/data/SWATGenXApp/GenXAppData/DEM/VPUID/{self.VPUID}/'
        streams_path = os.path.join(f'/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{self.VPUID}/streams.pkl')
        streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
        
        # Extract the UTM zone
        zone = streams.crs.to_string().split(' ')[1].split('=')[-1]
        epsg_code = int(f"326{zone[-2:]}")
        self.EPSG = epsg_code
        

    def reprojection(self, path, EPSG=32613):
        from osgeo import gdal, osr
        import os 
        raster = gdal.Open(path)
        output = path.replace(".tif", "_reprojected.tif")
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(EPSG)
        warp_options = gdal.WarpOptions(dstSRS=target_crs, format="GTiff", resampleAlg="near", creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
        gdal.Warp(output, raster, options=warp_options)
        ## remvoe original file
        os.remove(path)
        ## rename the reprojected file
        os.rename(output, path)


    def NLCD_extract_by_VPUID(self):
        NLCD_all_files = os.listdir(self.NLCD_all_files_path)
        for file in NLCD_all_files:
            if file.endswith(".img") and str(self.epoch) == file.split("_")[1]:
                self.NLCD_name = file
                self.NLCD_path = os.path.join(self.NLCD_all_files_path, self.NLCD_name)
                break

        raster = gdal.Open(self.NLCD_path)
        self.spatial_ref = raster.GetProjection()
        print(self.spatial_ref)  ## output: Albers_Conical_Equal_Area

        HUC4_name = os.listdir(self.HUC4_base_path)
        for name in HUC4_name:
            if name.endswith(".gdb"):
                self.HUC4_path = os.path.join(self.HUC4_base_path, name)
                break
        print(self.HUC4_path)

        self.HUC4 = gpd.read_file(self.HUC4_path, layer='WBDHU4')
        self.HUC4_crs = self.HUC4.crs
        print(self.HUC4_crs)  ## output: EPSG:4269

        self.HUC4 = self.HUC4.to_crs(self.spatial_ref)
        print(self.HUC4.crs)  ## output: Albers_Conical_Equal_Area

        ## now create a rectangular object for clipping the raster
        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=self.spatial_ref)

        os.makedirs(os.path.dirname(self.temp_path), exist_ok=True)
        rect.to_file(self.temp_path)

        ### now use the rect to clip the raster
        src_raster = gdal.Open(self.NLCD_path)
        rect_ds = ogr.Open(self.temp_path)
        rect_layer = rect_ds.GetLayer()

        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(self.original_out_raster, src_raster.RasterXSize, src_raster.RasterYSize, 1, gdal.GDT_Float32)
        gdal.Warp(self.original_out_raster, src_raster, cutlineDSName=self.temp_path, cropToCutline=True)

        DEM_path = self.get_DEM_path()
        target_crs = self.get_DEM_crs(DEM_path)
        self.get_EPSG()
        ### now project the original_out_raster to the target_crs and save it
        output_path = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{self.VPUID}/NLCD_{self.VPUID}_{self.epoch}_30m.tif"
        gdal.Warp(output_path, self.original_out_raster, dstSRS=target_crs)

        self.reprojection(output_path, EPSG=self.EPSG)

        ### now resample the raster to 100m, 200m, 500m, 1000m, 2000m
        resolutions = [100, 250, 500, 1000, 2000]
        for resolution in resolutions:
            output_path = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{self.VPUID}/NLCD_{self.VPUID}_{self.epoch}_{resolution}m.tif"
            if not os.path.exists(output_path):
                gdal.Warp(output_path, self.original_out_raster, xRes=resolution, yRes=resolution, resampleAlg=gdal.GRA_NearestNeighbour)
                self.reprojection(output_path, EPSG=self.EPSG)

# Usage
def NLCD_extract_by_VPUID(VPUID, epoch):
    original_out_raster = fr"/data/SWATGenXApp/GenXAppData/LandUse/NLCD_CONUS/{VPUID}/NLCD_{VPUID}_{epoch}.tif"
    if not os.path.exists(original_out_raster):
        extractor = NLCDExtractor(VPUID, epoch)
        extractor.NLCD_extract_by_VPUID()
    else:
        print(f"NLCD data for {VPUID} and {epoch} already exists.")
if __name__ == "__main__":
    VPUID = "1206"
    epoch = 2021
    NLCD_extract_by_VPUID(VPUID, epoch)