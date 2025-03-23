"""
Module for centralized path handling in MODGenX.
This module provides functions to generate consistent file paths
for various resources used throughout the application.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Union
import pathlib

class PathHandler:
    """
    A class to handle path generation for MODGenX applications.
    Centralizes all path definitions and provides methods to access them.
    """
    
    def __init__(self, config):
        """
        Initialize the PathHandler with configuration parameters.
        
        Parameters:
        -----------
        config : MODFLOWGenXPaths
            Configuration object containing path components
        """
        self.config = config
        
        # Base directory structure
        self.ROOT_DIR = "/data/SWATGenXApp"
        self.CODES_DIR = os.path.join(self.ROOT_DIR, "codes")
        self.GENXAPP_DATA_DIR = os.path.join(self.ROOT_DIR, "GenXAppData")
        self.USERS_DIR = os.path.join(self.ROOT_DIR, "Users")
        self.BIN_DIR = os.path.join(self.CODES_DIR, "bin")
        
        # Generated base paths
        if config.BASE_PATH is None:
            config.BASE_PATH = self.GENXAPP_DATA_DIR
        
        # User directories
        self.user_dir = os.path.join(self.USERS_DIR, config.username)
        self.vpuid_dir = os.path.join(self.user_dir, f"SWATplus_by_VPUID/{config.VPUID}")
        self.level_dir = os.path.join(self.vpuid_dir, config.LEVEL)
        self.name_dir = os.path.join(self.level_dir, config.NAME)
        
        # Model directories
        self.model_dir = os.path.join(self.name_dir, config.MODFLOW_MODEL_NAME)
        self.rasters_input_dir = os.path.join(self.model_dir, "rasters_input")
        
        # SWAT model directories
        self.swat_dir = os.path.join(self.name_dir, config.SWAT_MODEL_NAME)
        self.swat_watershed_dir = os.path.join(self.swat_dir, "Watershed")
        self.swat_shapes_dir = os.path.join(self.swat_watershed_dir, "Shapes")
        self.swat_rasters_dir = os.path.join(self.swat_watershed_dir, "Rasters")
        
        # Logging directory
        self.logs_dir = os.path.join(self.CODES_DIR, "MODFLOW", "logs")
        
        # Create frequently used paths
        self.dem_path = os.path.join(self.name_dir, f"DEM_{config.RESOLUTION}m.tif")
        
        # Ensure critical directories exist
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure critical directories exist"""
        directories = [
            self.rasters_input_dir,
            self.logs_dir,
            self.model_dir,
            os.path.dirname(self.dem_path)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Also ensure the temp directory exists
        os.makedirs(os.path.join(self.CODES_DIR, "MODFLOW", "MODGenX", "_temp"), exist_ok=True)
    
    def get_modflow_exe_path(self) -> str:
        """Get path to the MODFLOW executable"""
        return os.path.join(self.BIN_DIR, "modflow-nwt")
    
    def get_model_path(self) -> str:
        """Get the MODFLOW model directory path"""
        return self.model_dir
    
    def get_raster_input_dir(self) -> str:
        """Get the rasters input directory"""
        return self.rasters_input_dir
    
    def get_log_path(self, name: str) -> str:
        """Get log file path with specified name"""
        return os.path.join(self.logs_dir, f"{name}.log")
    
    def get_swat_shapefile(self, name: str) -> str:
        """Get path to a SWAT shapefile by name"""
        return os.path.join(self.swat_shapes_dir, f"{name}.shp")
    
    def get_swat_dem_path(self) -> str:
        """Get SWAT DEM path"""
        return os.path.join(self.swat_rasters_dir, "DEM", "dem.tif")
    
    def get_ref_raster_path(self) -> str:
        """Get reference raster path"""
        return self.dem_path
    
    def get_output_file(self, filename: str) -> str:
        """Get full path for an output file in the model directory"""
        return os.path.join(self.model_dir, filename)
    
    def get_raster_input_file(self, filename: str) -> str:
        """Get full path for a raster input file"""
        return os.path.join(self.rasters_input_dir, filename)
    
    def get_bound_raster_path(self) -> str:
        """Get bound raster path"""
        return os.path.join(self.rasters_input_dir, "bound.tif")
    
    def get_domain_raster_path(self) -> str:
        """Get domain raster path"""
        return os.path.join(self.rasters_input_dir, "domain.tif")
        
    def get_genxapp_data_path(self, *subdirs) -> str:
        """Get path within the GenXAppData directory"""
        return os.path.join(self.GENXAPP_DATA_DIR, *subdirs)
    
    def get_all_rasters_dir(self) -> str:
        """Get directory containing all rasters"""
        return os.path.join(self.GENXAPP_DATA_DIR, "all_rasters")
    
    def get_raster_paths(self, ML: bool) -> Dict[str, str]:
        """
        Generate paths for all required rasters.
        
        Parameters:
        -----------
        ML : bool
            Whether to use machine learning prediction rasters
            
        Returns:
        --------
        Dict[str, str]
            Dictionary of raster names to their file paths
        """
        rasters_dir = self.get_all_rasters_dir()
        resolution = self.config.RESOLUTION
        
        # Base paths that don't change between ML and non-ML
        paths = {
            "DEM": os.path.join(rasters_dir, f"DEM_{resolution}m.tif"),
            "recharge_data": os.path.join(rasters_dir, f"Recharge_{resolution}m.tif"),
            "SWL": os.path.join(rasters_dir, f"kriging_output_SWL_{resolution}m.tif"),
            "SWL_er": os.path.join(rasters_dir, f"kriging_stderr_SWL_{resolution}m.tif"),
        }
        
        # ML-specific or non-ML paths
        prefix = "predictions_ML" if ML else "kriging_output"
        err_prefix = "predictions_ML" if ML else "kriging_stderr"
        suffix = "" if ML else "m"
        
        # Define parameter mapping with correct keys
        parameter_mapping = {
            "H_COND_1": "k_horiz_1",
            "H_COND_2": "k_horiz_2",
            "V_COND_1": "k_vert_1", 
            "V_COND_2": "k_vert_2",
            "AQ_THK_1": "thickness_1",
            "AQ_THK_2": "thickness_2"
        }
        
        # Add parameter-specific paths with correct keys
        for param, key in parameter_mapping.items():
            paths[key] = os.path.join(rasters_dir, f"{prefix}_{param}_{resolution}{suffix}.tif")
            paths[f"{key}_er"] = os.path.join(rasters_dir, f"{err_prefix}_{param}_{resolution}{suffix}.tif")
        
        return paths
    
    def get_shapefile_paths(self) -> Dict[str, str]:
        """
        Generate paths for shapefiles.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary of shapefile names to their file paths
        """
        return {
            "lakes": os.path.join(self.swat_shapes_dir, "SWAT_plus_lakes.shp"),
            "rivers": os.path.join(self.swat_shapes_dir, "rivs1.shp"),
            "subbasins": os.path.join(self.swat_shapes_dir, "SWAT_plus_subbasins.shp"),
            "subs": os.path.join(self.swat_shapes_dir, "subs1.shp"),
            "grids": os.path.join(self.model_dir, "Grids_MODFLOW.geojson"),
        }
    
    def get_database_file_paths(self) -> Dict[str, str]:
        """
        Generate paths for database files.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary of database file names to their file paths
        """
        return {
            "COUNTY": self.get_genxapp_data_path("Well_data_krigging/Counties_dis_gr.geojson"),
            "huc12": self.get_genxapp_data_path("NHDPlusData/WBDHU12/WBDHU12_26990.geojson"),
            "huc8": self.get_genxapp_data_path("NHDPlusData/WBDHU8/WBDHU8_26990.geojson"),
            "huc4": self.get_genxapp_data_path("NHDPlusData/WBDHU4/WBDHU4_26990.geojson"),
            "streams": self.get_genxapp_data_path("NHDPlusData/streams.pkl"),
            "observations": self.get_genxapp_data_path("observations/observations_original.geojson"),
        }
    
    def get_temporary_path(self, filename: str) -> str:
        """Get a temporary file path"""
        return os.path.join(self.model_dir, f"_temp_{filename}")
    
    def create_load_raster_args(self) -> Dict:
        """
        Create a dictionary of arguments for load_raster.
        
        Returns:
        --------
        Dict
            Dictionary of arguments for load_raster function
        """
        return {
            'LEVEL': self.config.LEVEL,
            'RESOLUTION': self.config.RESOLUTION,
            'NAME': self.config.NAME,
            'ref_raster': self.get_ref_raster_path(),
            'bound_raster': self.get_bound_raster_path(), 
            'active': self.get_domain_raster_path(),
            'MODEL_NAME': self.config.MODFLOW_MODEL_NAME,
            'SWAT_MODEL_NAME': self.config.SWAT_MODEL_NAME,
            'username': self.config.username,
            'VPUID': self.config.VPUID,
            'config': self.config,
            'path_handler': self
        }
