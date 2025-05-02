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
import pandas as pd
from shapely.geometry import box
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths


def check_NLCD_by_VPUID(VPUID, epoch):
    paths = SWATGenXPaths()
    NLCD_path = paths.construct_path(paths.NLCD_path)
    RESOLUTIONS = [30, 100, 250, 500, 1000, 2000]
    for resolution in RESOLUTIONS:
        nlcd_file = paths.construct_path(NLCD_path, VPUID, f"NLCD_{VPUID}_{epoch}_{resolution}m.tif")
        if not os.path.exists(nlcd_file):
            raise ValueError(f"{nlcd_file} does not exist")
    print(f"#### NLCD data exists for {VPUID} #####")

class NLCDExtractor:
    def __init__(self, VPUID, epoch):
        self.VPUID = VPUID
        self.epoch = epoch
        self.paths = SWATGenXPaths()
        self.database_dir = self.paths.construct_path(self.paths.database_dir)
        self.extracted_nhd_swatplus_path = self.paths.construct_path(self.paths.extracted_nhd_swatplus_path)
        self.NLCD_path = self.paths.construct_path(self.paths.NLCD_path)
        self.DEM_path = self.paths.construct_path(self.paths.DEM_path)
        self.NLCD_release_path = self.paths.construct_path(self.paths.NLCD_release_path)
        self.extracted_nhd_path = self.paths.construct_path(self.extracted_nhd_swatplus_path, VPUID, "unzipped_NHDPlusVPU")
        self.temp_path = self.paths.construct_path(self.NLCD_path, VPUID, "rect.shp")
        self.original_out_raster = self.paths.construct_path(self.NLCD_path, VPUID, f"NLCD_{VPUID}_{epoch}.tif")
        self.VPUID_DEM_base = self.paths.construct_path(self.DEM_path, "VPUID", VPUID)
        self.NLCD_file_path = None
        self.spatial_ref = None
        self.HUC4_path = None
        self.HUC4 = None
        self.HUC4_crs = None

    def get_DEM_path(self):
        DEM_name = os.listdir(self.VPUID_DEM_base)
        for file in DEM_name:
            if file.endswith(".tif") and "30m" in file:
                return os.path.join(self.VPUID_DEM_base, file)
        raise FileNotFoundError("No DEM file with 30m resolution found.")

    def get_DEM_crs(self, DEM_path):
        raster = gdal.Open(DEM_path)
        return raster.GetProjection()

    def get_EPSG(self):
        streams_path = self.paths.construct_path(self.extracted_nhd_swatplus_path, self.VPUID, "streams.pkl")
        streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
        zone = streams.crs.to_string().split(' ')[1].split('=')[-1]
        self.EPSG = int(f"326{zone[-2:]}")

    def reprojection(self, path, EPSG):
        raster = gdal.Open(path)
        output = path.replace(".tif", "_reprojected.tif")
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(EPSG)
        warp_options = gdal.WarpOptions(dstSRS=target_crs, format="GTiff", resampleAlg="near", creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
        gdal.Warp(output, raster, options=warp_options)
        os.remove(path)
        os.rename(output, path)

    def NLCD_extract_by_VPUID(self):
        print(f"Extracting NLCD data for {self.VPUID} and {self.epoch}")
        self.NLCD_file_path = os.path.join(self.NLCD_release_path, f"Annual_NLCD_LndCov_{self.epoch}_CU_C1V0.tif")
        print(f"NLCD file path: {self.NLCD_file_path}")
        if not os.path.exists(self.NLCD_file_path):
            raise FileNotFoundError(f"{self.NLCD_file_path} does not exist")
        print(f"NLCD file path exists")
        raster = gdal.Open(self.NLCD_file_path)
        self.spatial_ref = raster.GetProjection()
        print(f"Spatial reference: {self.spatial_ref}")
        HUC4_name = os.listdir(self.extracted_nhd_path)
        for name in HUC4_name:
            if name.endswith(".gdb"):
                self.HUC4_path = os.path.join(self.extracted_nhd_path, name)
                print(f"HUC4 path: {self.HUC4_path}")
                break

        self.HUC4 = gpd.read_file(self.HUC4_path, layer='WBDHU4')
        print(f"HUC4: {self.HUC4}")
        self.HUC4 = self.HUC4.to_crs(self.spatial_ref)
        print(f"HUC4 CRS: {self.HUC4.crs}")

        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=self.spatial_ref)
        os.makedirs(os.path.dirname(self.temp_path), exist_ok=True)
        rect.to_file(self.temp_path)
        print(f"Rect: {rect}")
        src_raster = gdal.Open(self.NLCD_file_path)
        print(f"Src raster: {src_raster}")
        gdal.Warp(self.original_out_raster, src_raster, cutlineDSName=self.temp_path, cropToCutline=True)
        print(f"Original out raster: {self.original_out_raster}")
        DEM_path = self.get_DEM_path()
        print(f"DEM path: {DEM_path}")
        target_crs = self.get_DEM_crs(DEM_path)
        print(f"Target CRS: {target_crs}")
        self.get_EPSG()
        print(f"EPSG: {self.EPSG}")
        output_path = self.paths.construct_path(self.NLCD_path, "VPUID", self.VPUID, f"NLCD_{self.VPUID}_{self.epoch}_30m.tif")
        print(f"Output path: {output_path}")
        gdal.Warp(output_path, self.original_out_raster, dstSRS=target_crs)
        assert os.path.exists(output_path), f"Output path does not exist: {output_path}"
        self.reprojection(output_path, EPSG=self.EPSG)

        resolutions = [100, 250, 500, 1000, 2000]
        for resolution in resolutions:
            output_path = self.paths.construct_path(self.NLCD_path, self.VPUID, f"NLCD_{self.VPUID}_{self.epoch}_{resolution}m.tif")
            if not os.path.exists(output_path):
                gdal.Warp(output_path, self.original_out_raster, xRes=resolution, yRes=resolution, resampleAlg=gdal.GRA_NearestNeighbour)
                print(f"Output path: {output_path}")
                self.reprojection(output_path, EPSG=self.EPSG)

def NLCD_extract_by_VPUID_helper(VPUID, epoch):
    paths = SWATGenXPaths()
    original_out_raster = paths.construct_path(paths.NLCD_path, "VPUID",VPUID, f"NLCD_{VPUID}_{epoch}.tif")
    if not os.path.exists(original_out_raster):
        extractor = NLCDExtractor(VPUID, epoch)
        extractor.NLCD_extract_by_VPUID()
    else:
        print(f"NLCD data for {VPUID} and {epoch} already exists.")


if __name__ == "__main__":
    VPUID = "0405"
    epoch = 2021
    NLCD_extract_by_VPUID_helper(VPUID, epoch)
