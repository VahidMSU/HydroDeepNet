try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
import numpy as np

from MODGenX.logger_singleton import get_logger

logger = get_logger()

def lakes_to_drain(swat_lake_raster_path, top, k_horiz, load_raster_args):
    """
    Identify lake drainage locations and calculate drainage conductances.
    
    Parameters:
    -----------
    swat_lake_raster_path : str
        Path to the SWAT lake raster
    top : numpy.ndarray
        Top elevation array
    k_horiz : list
        List of horizontal hydraulic conductivity arrays
    load_raster_args : dict
        Dictionary containing parameters for raster loading with path_handler
    
    Returns:
    --------
    list
        List of tuples containing layer, row, column, elevation, and conductance for each drainage cell.
    """
    try:
        # Ensure path_handler is provided
        assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
        path_handler = load_raster_args['path_handler']
        ref_raster_path = path_handler.get_ref_raster_path()
        fit_to_meter = path_handler.config.fit_to_meter
        log_dir = os.path.dirname(path_handler.get_log_path("lakes"))
        
        logger.info(f"Using fit_to_meter value from config: {fit_to_meter}")
        
        # Load the lake raster and match its dimensions with the top elevation grid
        lake_raster = load_raster(swat_lake_raster_path, load_raster_args)
        lake_raster = match_raster_dimensions(load_raster(ref_raster_path, load_raster_args), lake_raster)

        # Handle infinite and NaN values
        lake_raster[lake_raster == np.inf] = 1
        lake_raster[lake_raster == 1] = 0

        # Identify drainage locations
        drain_loc = np.where(lake_raster != 0)

        # Initialize drain parameters
        drain_layer = np.full(drain_loc[0].shape, 0)
        drain_row = drain_loc[0]
        drain_col = drain_loc[1]
        
        # Apply fit_to_meter for consistent unit handling
        drain_elevations = top[drain_loc] - 1
        drain_conductances = k_horiz[0][drain_loc] * lake_raster[drain_loc] * fit_to_meter

        logger.info(f"Generated {len(drain_row)} drain cells for lakes with conductance range: {np.min(drain_conductances):.2f} to {np.max(drain_conductances):.2f}")

        return list(
            zip(
                drain_layer,
                drain_row,
                drain_col,
                drain_elevations,
                drain_conductances,
            )
        )
    except FileNotFoundError as e:
        logger.error(f"Lake raster file not found: {swat_lake_raster_path}")
        return []
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Error reading lake raster: {str(e)}")
        return []
    except ValueError as e:
        logger.error(f"Value error processing lake raster: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in lakes_to_drain: {str(e)}")
        return []
