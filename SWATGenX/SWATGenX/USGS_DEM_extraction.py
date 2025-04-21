# The `DEMProcessor` class in the provided Python script handles processing digital elevation model
# (DEM) data for a specific VPUID area.

import itertools
import geopandas as gpd
import requests
import rasterio
from rasterio.merge import merge
import glob
import math
import os
import utm
import os
import multiprocessing
import time
import sys
from osgeo import gdal, ogr
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import get_all_VPUIDs
except Exception:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from utils import get_all_VPUIDs
from osgeo import gdal, osr
class DEMProcessor:
    def __init__(self, VPUID, download_path=None, output_mosaic_path=None, url_download_path = None):
        self.VPUID = VPUID
        self.download_path = download_path
        self.output_mosaic_path = output_mosaic_path

        self.url_download_path = url_download_path

    def get_utm_zone(self, lat, lon):
        _, _, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return f"EPSG:326{zone_number}" if zone_letter >= 'N' else f"EPSG:327{zone_number}"

    def VPUID_HUC4_bounds(self, temp_path, utm_zone, unzipped_nhdplus_base_path):

        gdb_base = unzipped_nhdplus_base_path
        print("gdb_base", gdb_base)
        gdb_file_name = os.listdir(gdb_base)
        gdb_file_name = next(file for file in gdb_file_name if file.endswith('.gdb'))
        print("gdb_file_name", gdb_file_name)
        gdb_path = os.path.join(gdb_base, gdb_file_name)
        print("gdb_path", gdb_path)
        layer_name = "WBDHU4"
        gdf = gpd.read_file(filename=gdb_path, layer=layer_name)
        print("crs", gdf.crs)
        crs_original = gdf.crs
        print("GDF columns", gdf.columns)

        gdf = gdf.to_crs(utm_zone)
        #gdf['geometry'] = gdf['geometry'].buffer(10000)  # Buffer of 500 units
        #gdf = gdf.to_crs(crs_original)
        # simplify the geometry
        #gdf['geometry'] = gdf['geometry'].simplify(0.01)
        ### make it a single polygon

        try:
            gdf.dissolve(by='huc4').reset_index(drop=True)[['geometry']].to_file(temp_path)
        except Exception:
            gdf.dissolve(by='HUC4').reset_index(drop=True)[['geometry']].to_file(temp_path)

        print("HUC4 bounds saved to", temp_path)
    def clip_vpuid_mosaic(self, input_raster, temp_path, output_raster):
        """
        Clips the DEM mosaic to the extent of the clipping layer using GDAL.

        This function retrieves the extent of the vector file defining the clipping boundary
        and uses it to clip the input raster. The output is a raster file clipped to the extent.

        Args:
            input_raster (str): Path to the input raster file.
            temp_path (str): Path to the shapefile or vector file defining the clipping boundary.
            output_raster (str): Path to the output clipped raster file.
        """
        if os.path.exists(output_raster):
            print(f"Output raster {output_raster} already exists, continue...")
            return
        clipping_layer = self._extracted_from_resampling_13(
            "Clipping the DEM mosaic to the extent of the boundary...",
            ogr,
            temp_path,
            'Unable to open vector file: ',
        )
        # Get the first layer of the vector file
        layer = clipping_layer.GetLayer()

        # Check if geometry is valid
        if not layer.GetFeatureCount():
            raise ValueError("No features found in the clipping geometry.")

        # Get the extent of the clipping layer
        extent = layer.GetExtent()  # Returns (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = extent
        ## create a bounding box

        import geopandas
        from shapely.geometry import box
        bbox = box(xmin, ymin, xmax, ymax)
        ## get the crs of the temp_path
        crs = geopandas.read_file(temp_path).crs
        ## create a geodataframe
        gdf = geopandas.GeoDataFrame(geometry=[bbox], crs=crs)
        gdf.to_file(temp_path)
        # Print the extent

        print(f"Clipping extent: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

        # Set options for gdal.Warp with temp_path as the clipping layer
        warp_options = gdal.WarpOptions(

            cutlineDSName=temp_path,
            cutlineLayer=str(layer.GetName()),
            cutlineWhere=None,
            cropToCutline=True,

            dstNodata=None,
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"]
        )

        # Perform the clipping
        gdal.Warp(
            destNameOrDestDS=output_raster,
            srcDSOrSrcDSTab=input_raster,
            options=warp_options
        )

        assert os.path.exists(output_raster), f"Clipping failed. Output raster not found: {output_raster}"

        print(f"Clipping complete. Output saved to {output_raster}")

    def project_clipped_raster(self,output_raster, utm_zone, dem_base_path, output_raster_projected):
        """
        Projects the clipped raster to the specified UTM zone using GDAL.

        :param output_raster: Path to the input clipped raster file.
        :param utm_zone: EPSG code in the format 'EPSG:<code>'.
        :param dem_base_path: Base path for DEM (not used here, but kept for consistency with original function).
        :param output_raster_projected: Path to the output projected raster file.
        """
        print("Projecting the clipped raster to the original UTM zone...")


        # Open the input raster
        src_ds = gdal.Open(output_raster)
        if not src_ds:
            raise FileNotFoundError(f"Unable to open raster file: {output_raster}")

        # Get the source spatial reference
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(src_ds.GetProjection())
        print("##### utm_zone", utm_zone)
        # Define the target spatial reference
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(utm_zone))

        # Perform the reprojection using gdal.Warp
        warp_options = gdal.WarpOptions(
            dstSRS=dst_srs.ExportToWkt(),
            resampleAlg="nearest",
            xRes=30,  # Specify cell size in x-direction
            yRes=30,  # Specify cell size in y-direction
        )

        gdal.Warp(
            destNameOrDestDS=output_raster_projected,
            srcDSOrSrcDSTab=src_ds,
            options=warp_options
        )

        print("Processing complete.")

    def delete_temp_files(self,temp_path, output_raster):
        """
        Deletes temporary files if they exist.

        :param temp_path: Path to the temporary file.
        :param output_raster: Path to the output raster file.
        """
        print("Deleting temporary files...")

        # Check and delete temp_path if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"{temp_path} deleted.")
            except Exception as e:
                print(f"Failed to delete {temp_path}: {e}")

        # Check and delete output_raster if it exists
        if os.path.exists(output_raster):
            try:
                os.remove(output_raster)
                print(f"{output_raster} deleted.")
            except Exception as e:
                print(f"Failed to delete {output_raster}: {e}")

    def get_EPSG(self, VPUID):
        import pandas as pd
        base_input_raster = f'{SWATGenXPaths.DEM_path}/VPUID/{VPUID}/'
        streams_path = os.path.join(f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{VPUID}/streams.pkl')
        streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
        zone = streams.crs.to_string().split(' ')[1].split('=')[-1]
        return f"326{zone[-2:]}"


    def get_watershed_bounds(self,unzipped_nhdplus_base_path):
        gdb_base = unzipped_nhdplus_base_path
        gdb_name = next(file for file in os.listdir(gdb_base) if file.endswith('.gdb'))
        path = os.path.join(gdb_base, gdb_name)
        gdf = gpd.read_file(path, layer="WBDHU12")
        extent = gdf.total_bounds
        min_lat = extent[1]
        max_lat = extent[3]
        min_lon = extent[0]
        max_lon = extent[2]
        return {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon,
        }

    def list_required_dems(self, bounds):
        dems_to_download = []
        base_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/"

        min_lat_expanded = math.floor(bounds['min_lat'])
        max_lat_expanded = math.ceil(bounds['max_lat'])
        min_lon_expanded = math.floor(bounds['min_lon'])
        max_lon_expanded = math.ceil(bounds['max_lon'])

        if max_lat_expanded - bounds['max_lat'] < 1:
            max_lat_expanded += 1
        if bounds['min_lon'] - min_lon_expanded < 1:
            min_lon_expanded -= 1

        lat_range = range(min_lat_expanded, max_lat_expanded)
        lon_range = range(min_lon_expanded, max_lon_expanded)

        for lat, lon in itertools.product(lat_range, lon_range):
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'w' if lon < 0 else 'e'
            tile_name = f"USGS_13_{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"
            dems_to_download.append(base_url + tile_name)

        return dems_to_download

    def download_dems(self, dems_to_download, url_download_path):
        urls_to_download = []
        for url in dems_to_download:
            print(os.path.basename(url))
            filename = os.path.basename(url)
            with open(url_download_path, "r") as file:
                lines = file.readlines()
                urls_to_download.extend(line[:-1] for line in lines if filename in line)
        print("USGS DEMs to download:", urls_to_download, sep="\n")
        downloaded_files = []
        for url in urls_to_download:
            filename = os.path.basename(url)
            path = self.download_path
            os.makedirs(path, exist_ok=True)
            path_to_download = os.path.join(path, filename)
            downloaded_files.append(path_to_download)

            print(f"Processing DEM file: {filename}")
            print(f"path_to_download path: {path_to_download}")
            print(f"URL: {url}")

            filename = os.path.basename(url)

            ## if filename exist in {SWATGenXPaths.DEM_path}/CONUS
            list_of_already_downloaded_files = os.listdir(f"{SWATGenXPaths.DEM_path}/CONUS")
            if filename in list_of_already_downloaded_files:
                print(f"{filename} already downloaded, continue...")
                ## copy to {SWATGenXPaths.DEM_path}/VPUID/
                import shutil
                shutil.copy(f"{SWATGenXPaths.DEM_path}/CONUS/{filename}", path_to_download)
                continue

            print(f"Downloading {filename}...")
            import time

            time.sleep(15)
            if os.path.exists(path_to_download):
                print(f"{filename} already downloaded, continue...")
                continue
            else:
                print(f"{filename} does not exist, download it...")

            response = requests.get(url)

            if response.status_code == 200:
                if os.path.exists(path_to_download):
                    print(f"{filename} exists, continue...")
                    continue
                with open(path_to_download, 'wb') as file:
                    file.write(response.content)
                print(f"Data retrieved successfully for {filename}.")
            else:
                print(f"Failed to retrieve data: {response.status_code}, {response.text}")
        print(f"downloaded_files: {downloaded_files}")
        return downloaded_files


    def create_mosaic(self, downloaded_files, output_mosaic_path):

        """
        Creates a mosaic from multiple DEM files using GDAL.

        :param downloaded_files: List of paths to DEM files to be mosaicked.
        :param output_mosaic_path: Path where the output mosaic file will be saved.
        """
        print("Creating mosaic from DEM files...")

        if os.path.exists(output_mosaic_path):
            print(f"Mosaic already exists at {output_mosaic_path}, continue...")
            return

        # Open all input files and add them to a list
        src_files_to_mosaic = []
        for dem in downloaded_files:
            src = gdal.Open(dem)
            if not src:
                raise FileNotFoundError(f"Unable to open raster file: {dem}")
            src_files_to_mosaic.append(src)
            print(f"DEM {dem} opened and appended to mosaic.")

        # Create a virtual raster (in-memory mosaic)
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=True)
        vrt = gdal.BuildVRT('/vsimem/temp_mosaic.vrt', src_files_to_mosaic, options=vrt_options)

        if mosaic := gdal.Translate(
            output_mosaic_path,
            vrt,
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'],  # Include BIGTIFF=YES
        ):
            print(f"Mosaic created successfully at {output_mosaic_path}")
        else:
            raise RuntimeError("Failed to create mosaic.")



    def print_progress(self, message):
        print(message)

    def process_dem(self,unzipped_nhdplus_base_path):
        self.print_progress("########### Getting watershed bounds ############")
        watershed_bounds = self.get_watershed_bounds(unzipped_nhdplus_base_path)

        self.print_progress("########### Listing required DEMs ############")
        dems_to_download = self.list_required_dems(watershed_bounds)

        self.print_progress("########### Downloading DEMs ############")
        downloaded_files = self.download_dems(dems_to_download, self.url_download_path)

        self.print_progress("########### Creating mosaic ############")
        self.create_mosaic(downloaded_files, self.output_mosaic_path)

        self.print_progress("########### Clipping mosaic ############")


    def resampling(self,RESOLUTION, output_raster_projected, output_resampled_raster_path):
        """
        Resamples a raster to the specified resolution using GDAL.

        :param RESOLUTION: Desired resolution (cell size) for the output raster (e.g., 30 for 30x30).
        :param output_raster_projected: Path to the input raster file to be resampled.
        :param output_resampled_raster_path: Path where the resampled raster will be saved.
        """
        src_ds = self._extracted_from_resampling_13(
            "Resampling the raster...",
            gdal,
            output_raster_projected,
            'Unable to open raster file: ',
        )
        # Get the current spatial reference and extent
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        x_min = geotransform[0]
        y_max = geotransform[3]
        x_size = src_ds.RasterXSize
        y_size = src_ds.RasterYSize

        # Calculate new raster dimensions
        x_res = RESOLUTION
        y_res = RESOLUTION
        x_pixels = int((x_size * geotransform[1]) / x_res)
        y_pixels = int((y_size * abs(geotransform[5])) / y_res)

        if resampled := gdal.Warp(
            output_resampled_raster_path,
            src_ds,
            format="GTiff",
            xRes=x_res,
            yRes=y_res,
            resampleAlg="near",  # Nearest neighbor resampling
            outputBounds=[
                x_min,
                y_max - (y_pixels * y_res),
                x_min + (x_pixels * x_res),
                y_max,
            ],
            creationOptions=["COMPRESS=LZW"],
        ):
            print(f"Resampling complete. Resampled raster saved to {output_resampled_raster_path}")
        else:
            raise RuntimeError("Resampling failed.")

    def _extracted_from_resampling_13(self, arg0, arg1, arg2, arg3):
        print(arg0)
        if result := arg1.Open(arg2):
            return result
        else:
            raise FileNotFoundError(f"{arg3}{arg2}")


    def clean_directory(self, path, not_delete_pattern):
        files = glob.glob(f"{path}/*")
        for file in files:
            # if DEM_ exists in the file name, do not delete it
            if not_delete_pattern in file:
                print(f"{file} is not deleted")
                continue
            os.remove(file)


def warrper_NHDPlus_DEM(VPUID) -> None:
    DEM_extract_by_VPUID(VPUID)


def DEM_extract_by_VPUID(VPUID) -> None:
    dem_base_path = f"{SWATGenXPaths.DEM_path}/VPUID"
    unzipped_nhdplus_base_path = f"{SWATGenXPaths.extracted_nhd_swatplus_path}/{VPUID}/unzipped_NHDPlusVPU/"
    download_path = fr"{SWATGenXPaths.DEM_path}/VPUID/{VPUID}"
    output_mosaic_path = f"{SWATGenXPaths.DEM_path}/VPUID/{VPUID}/Mosaic.tif"
    temp_path = os.path.join(dem_base_path, f"{VPUID}/temp.shp")
    input_raster = os.path.join(dem_base_path, f"{VPUID}/Mosaic.tif")
    output_raster = os.path.join(dem_base_path, f"{VPUID}/Mosaic_clip.tif")
    url_download_path= os.path.join(dem_base_path, "DEM_13_arc_second.USGS")
    output_raster_projected = os.path.join(dem_base_path, f"{VPUID}/USGS_DEM_30m.tif")
    dem_processor = DEMProcessor(VPUID, download_path, output_mosaic_path,url_download_path)
    EPSG = dem_processor.get_EPSG(VPUID)

    if not os.path.exists(output_raster_projected):

        dem_processor.process_dem(unzipped_nhdplus_base_path)
        dem_processor.VPUID_HUC4_bounds(temp_path, EPSG, unzipped_nhdplus_base_path)
        dem_processor.clip_vpuid_mosaic(input_raster, temp_path, output_raster)
        dem_processor.project_clipped_raster(output_raster, EPSG, dem_base_path, output_raster_projected)
        dem_processor.delete_temp_files(temp_path, output_raster)

        RESOLUTIONS = [100, 250, 500, 1000, 2000]

        for RESOLUTION in RESOLUTIONS:

            output_resampled_raster_path = os.path.join(dem_base_path, f"{VPUID}/USGS_DEM_{RESOLUTION}m.tif")
            if not os.path.exists(output_resampled_raster_path):
                dem_processor.resampling(RESOLUTION, output_raster_projected,output_resampled_raster_path)


    import pandas as pd
    base_input_raster = f'{SWATGenXPaths.DEM_path}/VPUID/{VPUID}/'
    streams_path = os.path.join(f'{SWATGenXPaths.extracted_nhd_swatplus_path}/{VPUID}/streams.pkl')
    input_raster = os.path.join(base_input_raster, "USGS_DEM_30m.tif")
    streams = gpd.GeoDataFrame(pd.read_pickle(streams_path))
    print('Loaded streams CRS',streams.crs)

    with rasterio.open(input_raster) as src:
        print('Loaded raster CRS',src.crs)
        EPSG = src.crs.to_string()

    if streams.crs.to_string().split(' ')[1].split('=')[-1] != EPSG.split(':')[-1][-2:]:
        print('streams crs:',streams.crs.to_string().split(' ')[1].split('=')[-1])
        print('DEM crs:',EPSG.split(':')[-1])
        raise ValueError("CRS of the DEM raster and the streams are different")
    print(f"DEM for VPUID {VPUID} is ready")
    not_delete_pattern = "USGS_DEM"
    dem_processor.clean_directory(download_path, not_delete_pattern)


def check_DEM_by_VPUID(VPUID):
    dem_base_path = "{SWATGenXPaths.DEM_path}/VPUID"
    RESOLUTIONS = ["30"]
    for RESOLUTION in RESOLUTIONS:

        output_raster_projected = os.path.join(dem_base_path, f"{VPUID}/USGS_DEM_{RESOLUTION}m.tif")
        if not os.path.exists(output_raster_projected):
            raise FileNotFoundError(f"USGS DEM {RESOLUTION}m for VPUID {VPUID} not found.")

    print(f"## USGS DEM for VPUID {VPUID} exists. #####")

if __name__ == "__main__":
    VPUIDs = get_all_VPUIDs()
    processes = []
    for VPUID in VPUIDs:
        try:
            check_DEM_by_VPUID(VPUID)
        except FileNotFoundError as e:
            print(f"DEM not found for {VPUID}. Processing...")

   # for i, VPUID in enumerate(VPUIDs):
   #     ## only if the first two letter is 02
   #     if VPUID != "1206":
   #         continue
   #     print(f"Starting process {i}", VPUID)   #
#
#        process = multiprocessing.Process(target=warrper_NHDPlus_DEM, args=(VPUID,))
#        processes.append(process)
#        process.start()
#        if i%10 == 0 and i != 0:
#            time.sleep(200)
#    for process in processes:
#        process.join()
#    print("All processes have been completed")
