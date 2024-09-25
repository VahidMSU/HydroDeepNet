# The `DEMProcessor` class in the provided Python script handles processing digital elevation model
# (DEM) data for a specific VPUID area.

import arcpy
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
arcpy.env.overwriteOutput = True
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
        print("Clipping the DEM mosaic to the watershed boundary... ")
        # Setup environment
        arcpy.env.workspace = os.path.dirname(input_raster)
        arcpy.env.extent = "MAXOF"
        arcpy.env.snapRaster = input_raster
        arcpy.env.cellSize = "MINOF"
        arcpy.env.overwriteOutput = True

        # Clip the raster
        arcpy.Clip_management(
            in_raster=input_raster,
            out_raster=output_raster,
            in_template_dataset=temp_path,
            clipping_geometry="ClippingGeometry",
            maintain_clipping_extent="NO_MAINTAIN_EXTENT",
            nodata_value=None,
        )

        print("Clipping complete. ")

    def project_clipped_raster(self, output_raster, utm_zone, dem_base_path, output_raster_projected):
        # project the clipped raster to the original UTM zone
        print("Projecting the clipped raster to the original UTM zone... ")
        arcpy.env.workspace = os.path.dirname(output_raster)
        EPSG = int(utm_zone.split(":")[1])

        arcpy.env.workspace = os.path.dirname(output_raster)
        arcpy.env.overwriteOutput = True
        arcpy.env.snapRaster = output_raster
        arcpy.env.overwriteOutput = True
        arcpy.ProjectRaster_management(
            in_raster=output_raster,
            out_raster=output_raster_projected,
            out_coor_system=arcpy.SpatialReference(EPSG),
            resampling_type="NEAREST",
            cell_size="30 30",

        )

        print("Processing complete.")

    def delete_temp_files(self, temp_path, output_raster):
        print("delete temp files... ")
        if os.path.exists(temp_path):
            arcpy.Delete_management(temp_path)
            print(f'{temp_path} deleted ')
        elif os.path.exists(output_raster):
            arcpy.Delete_management(output_raster)

            print(f'{output_raster} deleted ')

    def get_EPSG(self, unzipped_nhdplus_base_path):
        watershed_bounds = self.get_watershed_bounds(unzipped_nhdplus_base_path)
        lat_center = (watershed_bounds["max_lat"] + watershed_bounds["min_lat"]) / 2
        lon_center = (watershed_bounds["max_lon"] + watershed_bounds["min_lon"]) / 2
        utm_zone = self.get_utm_zone(lat_center, lon_center)
        return int(utm_zone.split(":")[1]), utm_zone

    def get_watershed_bounds(self,unzipped_nhdplus_base_path):
        gdb_base = unzipped_nhdplus_base_path
        gdb_name = next(file for file in os.listdir(gdb_base) if file.endswith('.gdb'))
        path = os.path.join(gdb_base, gdb_name)
        arcpy.env.workspace = path
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
        return downloaded_files

    def create_mosaic(self, downloaded_files):
        dem_files = downloaded_files
        src_files_to_mosaic = []

        for dem in dem_files:
            src = arcpy.Raster(dem)
            src_files_to_mosaic.append(src)
            print(f"DEM {dem} opened and appended to mosaic.")

        mosaic = arcpy.MosaicToNewRaster_management(src_files_to_mosaic, os.path.dirname(self.output_mosaic_path), os.path.basename(self.output_mosaic_path), pixel_type="32_BIT_FLOAT", number_of_bands=1)

        print(f"Mosaic created successfully at {self.output_mosaic_path}")


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
        self.create_mosaic(downloaded_files)

        self.print_progress("########### Clipping mosaic ############")

    def resampling(self, RESOLUTION, output_raster_projected,output_resampled_raster_path):
        print("Resampling the raster... ")
        
        
        arcpy.Resample_management(
            in_raster=output_raster_projected,
            out_raster=output_resampled_raster_path,
            cell_size=f"{RESOLUTION} {RESOLUTION}",
            resampling_type="NEAREST"
        )
        print(f"Resampling complete. Resampled raster saved to {output_resampled_raster_path}")
        
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
    dem_base_path = "/data/MyDataBase/SWATGenXAppData/DEM/VPUID"
    unzipped_nhdplus_base_path = os.path.join("/data/MyDataBase/SWATGenXAppData/NHDPlusData",f"SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/")
    download_path = fr"/data/MyDataBase/SWATGenXAppData/DEM/VPUID/{VPUID}"
    output_mosaic_path = f"/data/MyDataBase/SWATGenXAppData/DEM/VPUID/{VPUID}/Mosaic.tif"
    temp_path = os.path.join(dem_base_path, f"{VPUID}/temp.shp")
    input_raster = os.path.join(dem_base_path, f"{VPUID}/Mosaic.tif")
    output_raster = os.path.join(dem_base_path, f"{VPUID}/Mosaic_clip.tif")
    url_download_path= os.path.join(dem_base_path, "DEM_13_arc_second.USGS")
    output_raster_projected = os.path.join(dem_base_path, f"{VPUID}/USGS_DEM_30m.tif")

    if not os.path.exists(output_raster_projected):
        dem_processor = DEMProcessor(VPUID, download_path, output_mosaic_path,url_download_path)

        EPSG, utm_zone = dem_processor.get_EPSG(unzipped_nhdplus_base_path)

        if not os.path.exists(output_raster_projected):
            
            dem_processor.process_dem(unzipped_nhdplus_base_path)    
            dem_processor.VPUID_HUC4_bounds(temp_path, utm_zone, unzipped_nhdplus_base_path)
            dem_processor.clip_vpuid_mosaic(input_raster, temp_path, output_raster)
            dem_processor.project_clipped_raster(output_raster, utm_zone, dem_base_path, output_raster_projected)
            dem_processor.delete_temp_files(temp_path, output_raster)

            RESOLUTIONS = [100, 250, 500, 1000, 2000]
            
            for RESOLUTION in RESOLUTIONS:
                
                output_resampled_raster_path = os.path.join(dem_base_path, f"{VPUID}/USGS_DEM_{RESOLUTION}m.tif")
                if not os.path.exists(output_resampled_raster_path):
                    dem_processor.resampling(RESOLUTION, output_raster_projected,output_resampled_raster_path)
            not_delete_pattern = "USGS_DEM"
            dem_processor.clean_directory(download_path, not_delete_pattern)
        else:
            print(f"{os.path.basename(output_raster_projected)} exists, continue...")


def get_all_VPUIDs(base_directory):
    
    path = os.path.join(base_directory, "NHDPlus_VPU_National")
    print("path", path) 
    files = [f for f in os.listdir(path) if f.endswith('.zip')]
    VPUIDs = [file.split('_')[2] for file in files]
    print(VPUIDs)
    return VPUIDs


#sys.path.insert(0, '/data/MyDataBase/SWATGenXAppData/codes')

if __name__ == "__main__":
    VPUIDs = get_all_VPUIDs("/data/MyDataBase/SWATGenXAppData/NHDPlusData")
    processes = []
    #VPUIDs = ['0202']
    print("VPUIDs", VPUIDs)
    for i, VPUID in enumerate(VPUIDs):
        ## only if the first two letter is 02
        if VPUID[:2] != "02":
            continue
        print(f"Starting process {i}", VPUID)   
        process = multiprocessing.Process(target=warrper_NHDPlus_DEM, args=(VPUID,))
        processes.append(process)
        process.start()
        if i%10 == 0 and i != 0:
            time.sleep(200)
    for process in processes:
        process.join()
    print("All processes have been completed")
    
    
