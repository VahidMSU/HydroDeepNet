# Import necessary modules to make them accessible from the package
# Import the logger modules first without initializing
from .logger_singleton import get_logger, initialize_logger

# Now import the rest of the modules
from .MODGenXCore import MODGenXCore
from .gdal_operations import gdal_sa
from .Logger import Logger

# Don't initialize logger here - let it be initialized when needed
# This avoids permission errors during import

#### indicate the version of the package
__version__ = "0.1.1"
__author__ = "Vahid Rafiei"
__email__ = "rafieiva@msu.edu"
__license__ = "GPL-2.0"
__description__ = "A package for generating MODFLOW models based on SWAT outputs"
__url__ = "HydroDeepNet.msu.edu"
__maintainer__ = "Vahid Rafiei"
