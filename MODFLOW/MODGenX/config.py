from dataclasses import dataclass
import os
#/data2/SWATGenXApp/codes/SWATGenX/SWATGenX/SWATGenXConfigPars.py
@dataclass
class MODFLOWGenXPaths:
    overwrite: bool = True
    username: str = None
    BASE_PATH: str = None
    report_path: str = None
    MODFLOW_MODEL_NAME: str = None
    SWAT_MODEL_NAME: str = None
    LEVEL: str = None
    VPUID: str = None
    NAME: str = None
    RESOLUTION: int = None
    moflow_exe_path = os.path.join("/data2/SWATGenXApp/codes/bin/", "MODFLOW-NWT_64.exe")
    EPSG = "EPSG:26990"
    dpi = 300
    top = None
    bound_raster_path = None
    domain_raster_path = None

    active = None

    def __post_init__(self):
        if self.username is not None and self.VPUID is not None and self.LEVEL is not None and self.NAME is not None and self.MODFLOW_MODEL_NAME is not None:
            
            
            self.BASE_PATH = f"/data2/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/"
            self.report_path: str = f"/data2/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/"


            self.MODFLOW_model_path = os.path.join(self.BASE_PATH, self.VPUID, self.LEVEL, self.NAME, self.MODFLOW_MODEL_NAME)
            self.raster_folder = os.path.join(self.MODFLOW_model_path, "rasters_input")
            self.swat_river_raster_path = os.path.join(self.MODFLOW_model_path, 'swat_river.tif')
            self.swat_lake_raster_path = os.path.join(self.MODFLOW_model_path, 'lake_raster.tif')
            self.head_of_last_time_step = os.path.join(self.MODFLOW_model_path, "head_of_last_time_step.jpeg")
            self.output_heads = os.path.join(self.MODFLOW_model_path, f"{self.MODFLOW_MODEL_NAME}.hds")
            self.out_shp = os.path.join(self.MODFLOW_model_path, "Grids_MODFLOW")


            self.SWAT_model_path = os.path.join(self.BASE_PATH, self.VPUID, self.LEVEL, self.NAME, self.SWAT_MODEL_NAME)

            
            self.swat_lake_shapefile_path = os.path.join(self.SWAT_model_path, "Watershed/Shapes/SWAT_plus_lakes.shp")
            self.ref_raster_path = os.path.join(self.SWAT_model_path, f"DEM_{self.RESOLUTION}m.tif")
            self.subbasin_path = os.path.join(self.SWAT_model_path, "Watershed/Shapes/subs1.shp")
            self.base_dem = os.path.join(self.SWAT_model_path, "Watershed/Rasters/DEM/dem.tif")
            self.shape_geometry = os.path.join(self.SWAT_model_path, "Watershed/Shapes/SWAT_plus_subbasins.shp")


            self.raster_path = os.path.join(self.raster_folder, f'{self.NAME}_DEM_{self.RESOLUTION}m.tif.tif')
            self.basin_path = os.path.join(self.raster_folder, 'basin_shape.shp')
            self.bound_path = os.path.join(self.raster_folder, 'bound_shape.shp')
            self.temp_image = os.path.join("codes/MODFLOW/MODGenX/_temp", f"{self.NAME}_{self.MODFLOW_MODEL_NAME}.jpeg")

        else:
            raise ValueError('username is required')