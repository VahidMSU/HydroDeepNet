try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
import numpy as np

from MODGenX.Logger import Logger

logger = Logger(verbose=True)

def lakes_to_drain(swat_lake_raster_path, top, k_horiz, load_raster_args):
    """
    Identify lake drainage locations and calculate drainage conductances.
    
    Parameters:
    ... (describe each parameter)
    
    Returns:
    List of tuples containing layer, row, column, elevation, and conductance for each drainage cell.
    """
    
    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    NAME = load_raster_args['NAME']
    ref_raster_path = load_raster_args['ref_raster']
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
    drain_elevations = top[drain_loc] - 1
    drain_conductances = k_horiz[0][drain_loc] * lake_raster[drain_loc]

    return list(
        zip(
            drain_layer,
            drain_row,
            drain_col,
            drain_elevations,
            drain_conductances,
        )
    )
