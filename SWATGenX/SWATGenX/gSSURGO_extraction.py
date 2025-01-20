import os
import geopandas as gpd
from shapely.geometry import box

import os
import geopandas as gpd
from osgeo import gdal, ogr, osr
from shapely.geometry import box
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except Exception:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
class SoilExtraction:
    def __init__(self, soil_configuration):
        self.VPUID = soil_configuration.get("VPUID")
        self.gSSURGO_raster = soil_configuration.get("gSSURGO_raster")
        self.spatial_ref = None
        self.HUC4 = None
        self.extent = None
        self.extracted_gSSURGO_path = None
        self.temp_path = None
        self.clipped_rasters = []
        self.out_raster = None
        self.target_crs = None
        self.original_RESOLUTION = 30
        self.resolutions = [100, 250, 500, 1000, 2000]
        self.gSSURGO_path = SWATGenXPaths.gSSURGO_path

    def print_raster_crs(self):
        raster = gdal.Open(self.gSSURGO_raster)
        self.spatial_ref = raster.GetProjection()
        print('Raster CRS:', self.spatial_ref)

    def read_HUC4_shapefile(self, HUC4_path):
        # Check available layers
        layers = gpd.read_file(HUC4_path, driver="FileGDB", layer=None)
        print("Available layers:", layers)
        self.HUC4 = gpd.read_file(HUC4_path, driver="FileGDB", layer="WBDHU4")
        print('HUC4 CRS before conversion:', self.HUC4.crs)

    def convert_HUC4_crs(self):
        self.HUC4 = self.HUC4.to_crs(self.spatial_ref)
        print('HUC4 CRS after conversion:', self.HUC4.crs)

    def get_HUC4_extent(self):
        self.extent = self.HUC4.total_bounds
        print('HUC4 extent after conversion:', self.extent)

    def create_rectangular_object(self):
        self.extracted_gSSURGO_path = fr"{self.gSSURGO_path}/{self.VPUID}/"
        os.makedirs(self.extracted_gSSURGO_path, exist_ok=True)
        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=self.spatial_ref)
        self.temp_path = fr"{self.gSSURGO_path}/{self.VPUID}/rect.shp"
        rect.to_file(self.temp_path)
        
    def clip_raster(self):
        self.out_raster = os.path.join(self.extracted_gSSURGO_path, f"soil_{self.VPUID}.tif")
        raster_paths = self.gSSURGO_raster if isinstance(self.gSSURGO_raster, list) else [self.gSSURGO_raster]

        for raster_path in raster_paths:
            print(f"Clipping {raster_path}...")
            if not os.path.exists(raster_path):
                print(f"Error: Raster file {raster_path} does not exist.")
                continue
            
            src_raster = gdal.Open(raster_path)
            if src_raster is None:
                print(f"Error: Unable to open raster file {raster_path}.")
                continue

            # Perform clipping and ensure the output raster matches the DEM CRS
            gdal.Warp(
                destNameOrDestDS=self.out_raster,
                srcDSOrSrcDSTab=src_raster,
                cutlineDSName=self.temp_path,
                cropToCutline=True,
                dstSRS=self.target_crs  # Match DEM CRS
            )
            print(f"Clipped raster saved to: {self.out_raster}")

    def project_raster(self):
        output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{self.original_RESOLUTION}m.tif")
        if not os.path.exists(output_path):
            src_ds = gdal.Open(self.out_raster)
            if src_ds is None:
                raise ValueError(f"Unable to open source raster: {self.out_raster}")

            # Ensure the target CRS is in WKT format
            srs = osr.SpatialReference()
            if srs.ImportFromWkt(self.target_crs) == 0:  # Check if conversion is successful
                target_crs_wkt = srs.ExportToWkt()
            else:
                raise ValueError(f"Invalid target CRS: {self.target_crs}")

            print(f"Projecting raster to DEM CRS: {target_crs_wkt}")
            
            # Perform the reprojection
            gdal.Warp(
                destNameOrDestDS=output_path,
                srcDSOrSrcDSTab=src_ds,
                dstSRS=target_crs_wkt  # Pass the CRS in WKT format
            )
            self.reprojection(output_path, EPSG=self.EPSG)

    
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

    def get_EPSG(self, VPUID):
        import pandas as pd 
        DEM_path = SWATGenXPaths.DEM_path
        extracted_NHD_path = SWATGenXPaths.extracted_nhd_swatplus_path
        
        base_input_raster = f'{DEM_path}/VPUID/{VPUID}/'
        streams_path = os.path.join(f'{extracted_NHD_path}/{VPUID}/streams.pkl')
        streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
        
        # Extract the UTM zone
        zone = streams.crs.to_string().split(' ')[1].split('=')[-1]
        epsg_code = int(f"326{zone[-2:]}")
        self.EPSG = epsg_code
        # Convert EPSG code to WKT using osr.SpatialReference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        self.target_crs = srs.ExportToWkt()  # Store the WKT format
        print(f"Extracted CRS for DEM (EPSG:{epsg_code}): {self.target_crs}")



    def resample_raster(self):
        for resolution in self.resolutions:
            output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{resolution}m.tif")
            gdal.Warp(output_path, self.out_raster, xRes=resolution, yRes=resolution, resampleAlg=gdal.GRA_NearestNeighbour)

            self.reprojection(output_path, EPSG=self.EPSG)

        print("Resampling is done.")


def gSSURGO_extract_by_VPUID(VPUID):
    gSSURGO_path = SWATGenXPaths.gSSURGO_path
    DEM_path = SWATGenXPaths.DEM_path
    if not os.path.exists(f"{gSSURGO_path}/{VPUID}/soil_{VPUID}.tif"):
        run(VPUID, DEM_path)
    else:
        print(f"gSSURGO Data already exists for {VPUID}")

def run(VPUID, DEM_path):
    
    gSSURGO_raster = SWATGenXPaths.gSSURGO_raster
    extracted_nhd_swatplus_path = SWATGenXPaths.extracted_nhd_swatplus_path
    HUC4_path = f"{extracted_nhd_swatplus_path}/{VPUID}/unzipped_NHDPlusVPU/"
    HUC4_name = os.listdir(HUC4_path)

    for name in HUC4_name:
        if name.endswith(".gdb"):
            HUC4_path = os.path.join(HUC4_path, name)
            break

    VPUID_DEM_base = f"{DEM_path}/VPUID/{VPUID}"

    soil_configuration = {
        "VPUID": VPUID,
        "gSSURGO_raster": gSSURGO_raster,
        "HUC4_path": HUC4_path,
        "VPUID_DEM_base": VPUID_DEM_base,
    }

    soil_extraction = SoilExtraction(soil_configuration)
    soil_extraction.print_raster_crs()
    soil_extraction.read_HUC4_shapefile(HUC4_path)
    soil_extraction.convert_HUC4_crs()
    soil_extraction.get_HUC4_extent()
    soil_extraction.create_rectangular_object()
    soil_extraction.clip_raster()
    soil_extraction.get_EPSG(VPUID)
    soil_extraction.project_raster()
    soil_extraction.resample_raster()
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import get_all_VPUIDs
except Exception:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from utils import get_all_VPUIDs


def gSSURGO_check_by_VPUID(VPUID):
    gSSURGO_path = SWATGenXPaths.gSSURGO_path
    RESOLUTIONS = [30, 100, 250, 500, 1000, 2000]
    for resolution in RESOLUTIONS:
        if not os.path.exists(f"{gSSURGO_path}{VPUID}/gSSURGO_{VPUID}_{resolution}m.tif"):
            raise ValueError(f"{gSSURGO_path}{VPUID}/gSSURGO_{VPUID}_{resolution}m.tif does not exist")
    print(f"### gSSURGO data exists for {VPUID} #####")


if __name__ == "__main__":
    VPUID = "1206"
    VPUIDs = get_all_VPUIDs()
    for VPUID in VPUIDs:
        try:
            gSSURGO_check_by_VPUID(VPUID)   
            print(f"NLCD data exists for {VPUID}")
        except Exception as e:
            #gSSURGO_extract_by_VPUID(VPUID)
            print(f"Error in {VPUID} {str(e)}")