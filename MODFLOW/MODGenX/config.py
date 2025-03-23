from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
import os

@dataclass
class MODFLOWGenXPaths:
    overwrite: bool = True
    username: Optional[str] = None
    BASE_PATH: Optional[str] = None
    report_path: Optional[str] = None
    MODFLOW_MODEL_NAME: Optional[str] = None
    SWAT_MODEL_NAME: Optional[str] = None
    LEVEL: Optional[str] = None
    VPUID: Optional[str] = None
    NAME: Optional[str] = None
    RESOLUTION: Optional[int] = None
    moflow_exe_path: str = field(default_factory=lambda: os.path.join("/data/SWATGenXApp/codes/bin/", "modflow-nwt"))
    EPSG: str = "EPSG:26990"
    dpi: int = 300
    top: Any = None
    bound_raster_path: Optional[str] = None
    domain_raster_path: Optional[str] = None
    # Model discretization parameters
    n_sublay_1: int = 2  # Number of sub-layers in the first main layer
    n_sublay_2: int = 3  # Number of sub-layers in the second main layer
    # Bedrock parameters
    k_bedrock: float = 1e-4  # Bedrock hydraulic conductivity (m/day)
    bedrock_thickness: float = 40  # Bedrock thickness (m)
    # Unit conversion factors
    fit_to_meter: float = 0.3048  # Feet to meters conversion
    recharge_conv_factor: float = 0.0254/365.25  # Inch/year to meter/day conversion
    # Model convergence parameters
    headtol: float = 0.01  # Head change tolerance for convergence (m)
    fluxtol: float = 0.001  # Flux change tolerance for convergence (m³/day)
    maxiterout: int = 100  # Maximum number of outer iterations

    # Enable string representation for easier debugging
    def __str__(self):
        """Return a string representation of the configuration."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"MODFLOWGenXPaths({', '.join(attrs)})"
    
    # Allow retrieval of attributes by string name
    def get(self, name, default=None):
        """Get an attribute by name with an optional default value."""
        return getattr(self, name, default)
    
    # Validate configuration
    def validate(self):
        """Validate configuration and set defaults for missing values."""
        if self.RESOLUTION is None:
            self.RESOLUTION = 250
        
        if self.MODFLOW_MODEL_NAME is None:
            self.MODFLOW_MODEL_NAME = f'MODFLOW_{self.RESOLUTION}m'
        
        if self.SWAT_MODEL_NAME is None:
            self.SWAT_MODEL_NAME = 'SWAT_MODEL_Web_Application'
        
        if self.LEVEL is None:
            self.LEVEL = 'huc12'
        
        if self.BASE_PATH is None:
            self.BASE_PATH = '/data/SWATGenXApp/GenXAppData/'
        
        # Validate model parameters
        if self.n_sublay_1 <= 0:
            self.n_sublay_1 = 2
        
        if self.n_sublay_2 <= 0:
            self.n_sublay_2 = 3
        
        if self.k_bedrock <= 0:
            self.k_bedrock = 1e-4
        
        if self.bedrock_thickness <= 0:
            self.bedrock_thickness = 40
        
        return self
