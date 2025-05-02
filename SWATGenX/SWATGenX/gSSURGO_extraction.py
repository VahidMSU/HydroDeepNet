import os
import geopandas as gpd
from osgeo import gdal, ogr, osr
from shapely.geometry import box
import numpy as np
import rasterio
from rasterio.windows import Window
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except Exception:
    from SWATGenXConfigPars import SWATGenXPaths

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
        """Print the coordinate reference system of the source raster."""
        with rasterio.open(self.gSSURGO_raster) as src:
            self.spatial_ref = src.crs
            print('Raster CRS:', self.spatial_ref)

    def read_HUC4_shapefile(self, HUC4_path):
        """Read the HUC4 shapefile and print available layers."""
        try:
            self.HUC4 = gpd.read_file(HUC4_path, driver="FileGDB", layer="WBDHU4")
            print('HUC4 CRS before conversion:', self.HUC4.crs)
        except Exception as e:
            print(f"Error reading HUC4 shapefile: {e}")
            raise

    def convert_HUC4_crs(self):
        """Convert HUC4 to match the source raster's CRS."""
        if self.spatial_ref:
            self.HUC4 = self.HUC4.to_crs(self.spatial_ref)
            print('HUC4 CRS after conversion:', self.HUC4.crs)

    def get_HUC4_extent(self):
        """Get the extent of the HUC4 area."""
        self.extent = self.HUC4.total_bounds
        print('HUC4 extent after conversion:', self.extent)

    def create_rectangular_object(self):
        """Create a rectangular shapefile for clipping."""
        self.extracted_gSSURGO_path = os.path.join(self.gSSURGO_path, "VPUID", self.VPUID)
        os.makedirs(self.extracted_gSSURGO_path, exist_ok=True)

        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=self.spatial_ref)
        self.temp_path = os.path.join(self.extracted_gSSURGO_path, "rect.shp")
        rect.to_file(self.temp_path)

    def clip_raster(self):
        """Clip the raster using GDAL Warp."""
        self.out_raster = os.path.join(self.extracted_gSSURGO_path, f"soil_{self.VPUID}.tif")

        # Ensure the source raster exists
        if not os.path.exists(self.gSSURGO_raster):
            raise FileNotFoundError(f"Source raster not found: {self.gSSURGO_raster}")

        # Use GDAL Warp for clipping
        warp_options = gdal.WarpOptions(
            cutlineDSName=self.temp_path,
            cropToCutline=True,
            dstSRS=self.spatial_ref,
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']
        )

        gdal.Warp(self.out_raster, self.gSSURGO_raster, options=warp_options)
        print(f"Clipped raster saved to: {self.out_raster}")

    def get_EPSG(self, VPUID):
        """Get the EPSG code for the target CRS."""
        extracted_NHD_path = SWATGenXPaths.extracted_nhd_swatplus_path
        streams_path = os.path.join(f'{extracted_NHD_path}/{VPUID}/streams.pkl')

        if not os.path.exists(streams_path):
            raise FileNotFoundError(f"Streams file not found: {streams_path}")

        # Read pickle file using pandas
        streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
        zone = streams.crs.to_string().split(' ')[1].split('=')[-1]
        epsg_code = int(f"326{zone[-2:]}")
        self.EPSG = epsg_code

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        self.target_crs = srs.ExportToWkt()
        print(f"Extracted CRS for DEM (EPSG:{epsg_code}): {self.target_crs}")

    def project_raster(self):
        """Project the clipped raster to the target CRS."""
        output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{self.original_RESOLUTION}m.tif")

        if not os.path.exists(self.out_raster):
            raise FileNotFoundError(f"Clipped raster not found: {self.out_raster}")

        warp_options = gdal.WarpOptions(
            dstSRS=self.target_crs,
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']
        )

        gdal.Warp(output_path, self.out_raster, options=warp_options)
        self.reprojection(output_path, EPSG=self.EPSG)

    def reprojection(self, path, EPSG=32613):
        """Reproject the raster to the specified EPSG."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input raster not found: {path}")

        output = path.replace(".tif", "_reprojected.tif")
        target_crs = osr.SpatialReference()
        target_crs.ImportFromEPSG(EPSG)

        warp_options = gdal.WarpOptions(
            dstSRS=target_crs,
            format="GTiff",
            resampleAlg="near",
            creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"]
        )

        gdal.Warp(output, path, options=warp_options)
        os.remove(path)
        os.rename(output, path)

    def resample_raster(self):
        """Resample the raster to different resolutions."""
        if not os.path.exists(self.out_raster):
            raise FileNotFoundError(f"Source raster not found: {self.out_raster}")

        for resolution in self.resolutions:
            output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{resolution}m.tif")

            warp_options = gdal.WarpOptions(
                xRes=resolution,
                yRes=resolution,
                resampleAlg="near",
                creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']
            )

            gdal.Warp(output_path, self.out_raster, options=warp_options)
            self.reprojection(output_path, EPSG=self.EPSG)

        print("Resampling completed.")

def gSSURGO_extract_by_VPUID(VPUID):
    """Main function to extract gSSURGO data for a given VPUID."""
    gSSURGO_raster = SWATGenXPaths.gSSURGO_raster
    extracted_nhd_swatplus_path = SWATGenXPaths.extracted_nhd_swatplus_path

    # Find the HUC4 GDB file
    HUC4_path = os.path.join(extracted_nhd_swatplus_path, VPUID, "unzipped_NHDPlusVPU")
    HUC4_name = [name for name in os.listdir(HUC4_path) if name.endswith(".gdb")][0]
    HUC4_path = os.path.join(HUC4_path, HUC4_name)

    soil_configuration = {
        "VPUID": VPUID,
        "gSSURGO_raster": gSSURGO_raster,
        "HUC4_path": HUC4_path
    }

    try:
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
        print(f"Successfully processed VPUID: {VPUID}")
    except Exception as e:
        print(f"Error processing VPUID {VPUID}: {str(e)}")
        raise

if __name__ == "__main__":
    VPUID = "0405"
    gSSURGO_extract_by_VPUID(VPUID)
