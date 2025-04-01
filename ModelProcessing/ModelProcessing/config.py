"""
Configuration module for the ModelProcessing package.

This module defines dataclasses that encapsulate the configuration parameters
used throughout the ModelProcessing components, reducing the need for
lengthy parameter lists in function calls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os


@dataclass
class ModelConfig:
    """Configuration parameters for SWAT model processing."""
    
    # Basic identification parameters
    username: str = ""
    VPUID: str = ""
    LEVEL: str = "huc12"
    NAME: str = ""
    MODEL_NAME: str = ""
    
    # Path configuration
    BASE_PATH: str = field(default="")
    
    # Time periods
    START_YEAR: int = 2000
    END_YEAR: int = 2010
    Ver_START_YEAR: int = 2011
    Ver_END_YEAR: int = 2015
    nyskip: int = 3
    Ver_nyskip: int = 1
    
    # Process flags
    sensitivity_flag: bool = False
    calibration_flag: bool = True
    verification_flag: bool = False
    range_reduction_flag: bool = False
    
    # Sensitivity analysis parameters
    sen_total_evaluations: int = 100
    sen_pool_size: int = 10
    num_levels: int = 4
    
    # Calibration parameters
    cal_pool_size: int = 24
    max_cal_iterations: int = 10
    termination_tolerance: int = 10
    epsilon: float = 0.01
    verification_samples: int = 5
    no_value: float = 9999
    
    # SWAT model parameters
    pet: int = 0
    cn: int = 2
    
    # Derived paths (calculated during initialization)
    TxtInOut: str = ""
    model_log_path: str = ""
    scenarios_path: str = ""
    directory_path_si: str = ""
    initial_points_path: str = ""
    initial_values_path: str = ""
    best_simulation_filename: str = ""
    cal_file_path: str = ""
    model_base: str = ""
    
    # Additional derived path variables (will be set in __post_init__)
    lake_path: str = ""
    monthly_cal_figures_path: str = ""
    daily_cal_figures_path: str = ""
    calibration_figures_path: str = ""
    monthly_sen_figures_path: str = ""
    daily_sen_figures_path: str = ""
    local_best_solutions_path: str = ""
    gis_folder: str = ""
    ver_perf_path: str = ""
    
    def __post_init__(self):
        """Initialize derived paths after the basic fields are set."""
        # Set BASE_PATH if it's empty
        if not self.BASE_PATH:
            self.BASE_PATH = f"/data/SWATGenXApp/Users/{self.username}"
            
        # Set model base path
        self.model_base = os.path.join(self.BASE_PATH, f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/')
        
        # Set log paths
        self.model_log_path = os.path.join(self.model_base, 'log.txt')
        
        # Set model directories
        self.TxtInOut = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios/Default/TxtInOut/')
        self.scenarios_path = os.path.join(self.model_base, f'{self.MODEL_NAME}/Scenarios')
        
        # Set analysis file paths
        self.directory_path_si = os.path.join(self.model_base, f'morris_Si_{self.MODEL_NAME}.csv')
        self.initial_points_path = os.path.join(self.model_base, f'initial_points_{self.MODEL_NAME}.csv')
        self.initial_values_path = os.path.join(self.model_base, f'initial_values_{self.MODEL_NAME}.csv')
        self.best_simulation_filename = os.path.join(self.model_base, f'best_solution_{self.MODEL_NAME}.txt')
        self.cal_file_path = os.path.join(self.model_base, f'cal_parms_{self.MODEL_NAME}.cal')
        
        # Set additional paths for figures and model components
        self.lake_path = os.path.join(self.model_base, f"{self.MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp")
        self.monthly_cal_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_monthly")
        self.daily_cal_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_calibration_daily")
        self.calibration_figures_path = os.path.join(self.model_base, f"calibration_figures_{self.MODEL_NAME}")
        self.monthly_sen_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_monthly")
        self.daily_sen_figures_path = os.path.join(self.model_base, f"figures_{self.MODEL_NAME}_sensitivity_daily")
        self.local_best_solutions_path = os.path.join(self.model_base, f'local_best_solution_{self.MODEL_NAME}.txt')
        self.gis_folder = os.path.join(self.model_base, f'{self.MODEL_NAME}/gwflow_gis')
        self.ver_perf_path = os.path.join(self.model_base, f'recharg_output_{self.MODEL_NAME}/verification_performance_{self.MODEL_NAME}.txt')