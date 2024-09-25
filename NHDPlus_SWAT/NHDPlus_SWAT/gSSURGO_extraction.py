import arcpy
import os
import geopandas as gpd
from shapely.geometry import box

class SoilExtraction:
    def __init__(self, VPUID, gSSURGO_path):
        self.VPUID = VPUID
        self.gSSURGO_path = gSSURGO_path
        self.gSSURGO_raster = None
        self.spatial_ref = None
        self.HUC4 = None
        self.extent = None
        self.extracted_gSSURGO_path = None
        self.temp_path = None
        self.clipped_rasters = []
        self.out_raster = None
        self.DEM_path = None
        self.target_crs = None
        self.original_RESOLUTION = 30
        self.resolutions = [100, 250, 500, 1000, 2000]
        arcpy.env.overwriteOutput = True
    def set_workspace(self):
        arcpy.env.workspace = self.gSSURGO_path

    def list_gSSURGO_raster(self):
        self.gSSURGO_raster = arcpy.ListRasters()
        print(self.gSSURGO_raster)

    def print_raster_crs(self):
        self.spatial_ref = arcpy.Describe(self.gSSURGO_raster[0]).spatialReference
        print('Raster CRS:', self.spatial_ref.name)

    def read_HUC4_shapefile(self, HUC4_path):
        self.HUC4 = gpd.read_file(HUC4_path, layer='WBDHU4')
        print('HUC4 CRS before conversion:', self.HUC4.crs)

    def convert_HUC4_crs(self):
        self.HUC4 = self.HUC4.to_crs(self.spatial_ref.name)
        print('HUC4 CRS after conversion:', self.HUC4.crs)

    def get_HUC4_extent(self):
        self.extent = self.HUC4.total_bounds
        print('HUC4 extent after conversion:', self.extent)

    def create_rectangular_object(self):
        self.extracted_gSSURGO_path = fr"/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_CONUS/{self.VPUID}/"
        os.makedirs(self.extracted_gSSURGO_path, exist_ok=True)
        bounds = self.HUC4.total_bounds
        rect = gpd.GeoDataFrame(geometry=[box(*bounds)])
        rect.crs = self.spatial_ref.name
        self.temp_path = fr"/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_CONUS/{self.VPUID}/rect.shp"
        rect.to_file(self.temp_path)

    def clip_raster(self):
        self.out_raster = os.path.join(self.extracted_gSSURGO_path, f"soil_{self.VPUID}.tif")
        for raster in self.gSSURGO_raster:
            print(f"Clipping {raster}...")
            if not os.path.exists(self.out_raster):
                print(f"Clipped {raster} exists.")
                arcpy.Clip_management(raster, self.temp_path, self.out_raster, "#", "#", "NONE")
                self.clipped_rasters.append(self.out_raster)

    def get_DEM_path(self, VPUID_DEM_base):
        DEM_name = os.listdir(VPUID_DEM_base)
        for file in DEM_name: ### 500m for easy crs extraction
            if file.endswith(".tif") and "500m" in file:
                self.DEM_path = os.path.join(VPUID_DEM_base, file)
                break

    def get_DEM_crs(self):
        self.target_crs = arcpy.Describe(self.DEM_path).spatialReference

    def project_raster(self):
        arcpy.env.outputCoordinateSystem = self.target_crs
        output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{self.original_RESOLUTION}m.tif")
        if not os.path.exists(output_path):
            arcpy.ProjectRaster_management(self.out_raster, output_path, self.target_crs)

    def resample_raster(self):
        arcpy.env.outputCoordinateSystem = self.target_crs
        for resolution in self.resolutions:
            output_path = os.path.join(self.extracted_gSSURGO_path, f"gSSURGO_{self.VPUID}_{resolution}m.tif")
            arcpy.Resample_management(self.out_raster, output_path, resolution, "NEAREST")

        print("Clipping is done.")

# Usage example:
def gSSURGO_extract_by_VPUID(VPUID):


    if not os.path.exists(f"/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_CONUS/{VPUID}/soil_{VPUID}.tif"):
        gSSURGO_path = "/data/MyDataBase/SWATGenXAppData/Soil/gSSURGO_CONUS/gSSURGO_CONUS.gdb"
        HUC4_path = fr"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
        HUC4_name = os.listdir(HUC4_path)
        for name in HUC4_name:
            if name.endswith(".gdb"):
                HUC4_path = os.path.join(HUC4_path, name)
                break

        VPUID_DEM_base = f"/data/MyDataBase/SWATGenXAppData/DEM/VPUID/{VPUID}"

        soil_extraction = SoilExtraction(VPUID, gSSURGO_path)
        soil_extraction.set_workspace()
        soil_extraction.list_gSSURGO_raster()
        soil_extraction.print_raster_crs()
        soil_extraction.read_HUC4_shapefile(HUC4_path)
        soil_extraction.convert_HUC4_crs()
        soil_extraction.get_HUC4_extent()
        soil_extraction.create_rectangular_object()
        soil_extraction.clip_raster()
        soil_extraction.get_DEM_path(VPUID_DEM_base)
        soil_extraction.get_DEM_crs()
        soil_extraction.project_raster()
        soil_extraction.resample_raster()
    else:
        print(f"gSSURGO Data already exists for {VPUID}")

if __name__ == "__main__":
    VPUID = "0407"
    gSSURGO_extract_by_VPUID(VPUID)