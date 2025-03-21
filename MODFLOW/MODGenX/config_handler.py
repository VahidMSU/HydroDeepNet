import os

class ConfigHandler:
    """
    Configuration handler for MODGenX module.
    Manages paths and settings consistently across all components.
    """
    
    def __init__(self, swatgenx_config):
        """
        Initialize with a SWATGenXPaths config object
        
        Parameters:
        - swatgenx_config: SWATGenXPaths configuration object
        """
        self.config = swatgenx_config
        
        # Core properties from config
        self.NAME = self.config.NAME
        self.BASE_PATH = self.config.base_path
        self.LEVEL = self.config.LEVEL
        self.RESOLUTION = self.config.RESOLUTION
        self.MODFLOW_MODEL_NAME = self.config.MODFLOW_MODEL_NAME
        self.SWAT_MODEL_NAME = self.config.SWAT_MODEL_NAME
        self.VPUID = self.config.VPUID
        
        # Common static paths
        self.swat_prefix = f"SWATplus_by_VPUID/{self.VPUID}"
        self.raster_dir = "all_rasters"
        self.bin_dir = "bin"
        
    def get_base_modflow_path(self):
        """Returns the base path for the MODFLOW model"""
        return os.path.join(self.BASE_PATH, self.swat_prefix, self.LEVEL, self.NAME, self.MODFLOW_MODEL_NAME)
    
    def get_swat_path(self):
        """Returns the base path for the SWAT model"""
        return os.path.join(self.BASE_PATH, self.swat_prefix, self.LEVEL, self.NAME, self.SWAT_MODEL_NAME)
    
    def get_raster_folder_path(self):
        """Returns the path for raster inputs"""
        return os.path.join(self.get_base_modflow_path(), "rasters_input")
    
    def get_model_exe_path(self):
        """Returns the path to the MODFLOW executable"""
        return os.path.join(self.get_base_modflow_path(), "MODFLOW-NWT_64.exe")
    
    def get_reference_dem_path(self):
        """Returns the path to the reference DEM"""
        return os.path.join(self.BASE_PATH, self.swat_prefix, self.LEVEL, self.NAME, f"DEM_{self.RESOLUTION}m.tif")
    
    def get_swat_shape_path(self, shape_name):
        """Returns the path to a SWAT shape file"""
        return os.path.join(self.get_swat_path(), "Watershed/Shapes", shape_name)
    
    def get_raster_path(self, raster_name):
        """Returns the path to a raster in the all_rasters directory"""
        return os.path.join(self.BASE_PATH, self.raster_dir, f"{raster_name}_{self.RESOLUTION}m.tif")
    
    def get_output_path(self, filename):
        """Returns a path in the model output directory"""
        return os.path.join(self.get_base_modflow_path(), filename)
    
    def get_model_grid_path(self):
        """Returns the path to the model grid shapefile"""
        return os.path.join(self.get_base_modflow_path(), "Grids_MODFLOW")
    
    def get_load_raster_args(self):
        """Returns a dictionary of load_raster_args to maintain compatibility"""
        ref_raster_path = self.get_reference_dem_path()
        bound_raster_path = os.path.join(self.get_raster_folder_path(), 'bound.tif')
        domain_raster_path = os.path.join(self.get_raster_folder_path(), 'domain.tif')
        
        return {
            'LEVEL': self.LEVEL,
            'RESOLUTION': self.RESOLUTION,
            'NAME': self.NAME,
            'ref_raster': ref_raster_path,
            'bound_raster': bound_raster_path,
            'active': domain_raster_path,
            'MODEL_NAME': self.MODFLOW_MODEL_NAME,
            'SWAT_MODEL_NAME': self.SWAT_MODEL_NAME,
            'VPUID': self.VPUID
        }
    
    def create_directories(self):
        """Creates all necessary directories for the model"""
        os.makedirs(self.get_base_modflow_path(), exist_ok=True)
        os.makedirs(self.get_raster_folder_path(), exist_ok=True)
