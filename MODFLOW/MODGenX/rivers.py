try:
    from MODGenX.utils import *
except ImportError:
    from utils import *
import numpy as np

from MODGenX.Logger import Logger

logger = Logger(verbose=True)   

def river_gen(nrow, ncol, swat_river, top, ibound, load_raster_args=None):
    """
    Generate river data based on given conditions.
    
    Parameters:
    -----------
    nrow : int
        Number of rows in the grid
    ncol : int
        Number of columns in the grid
    swat_river : 2D array
        Array containing SWAT river data
    top : 2D array
        Array containing top elevation data
    ibound : 3D array
        Array containing cell activity data
    load_raster_args : dict, optional
        Dictionary containing parameters for raster loading, including config
    
    Returns:
    --------
    dict
        Dictionary containing information about river cells
    """
    # Get fit_to_meter from config if available
    fit_to_meter = 0.3048  # default value
    if load_raster_args and 'config' in load_raster_args and hasattr(load_raster_args['config'], 'fit_to_meter'):
        fit_to_meter = load_raster_args['config'].fit_to_meter
        logger.info(f"Using fit_to_meter value from config: {fit_to_meter}")
    
    # Get log directory for diagnostics
    log_dir = "/data/SWATGenXApp/codes/MODFLOW/logs"
    if load_raster_args and 'path_handler' in load_raster_args:
        try:
            log_dir = os.path.dirname(load_raster_args['path_handler'].get_log_path("river_gen"))
        except Exception:
            pass  # Use default log_dir
    
    # Validate input shapes
    assert top.shape == swat_river.shape, f"Shape mismatch: top {top.shape} vs swat_river {swat_river.shape}"
    
    # Use vectorized operations for better performance
    # Find river cells (where swat_river > 0 and top > 0)
    river_mask = (swat_river > 0) & (top > 0)
    rows, cols = np.where(river_mask)
    
    # Calculate river parameters for these cells
    stages = top[river_mask] + 1
    conds = swat_river[river_mask] * fit_to_meter
    bottoms = top[river_mask] - 1
    
    # Create river cells list
    river_data = {0: []}
    for i, (r, c, stage, cond, bottom) in enumerate(zip(rows, cols, stages, conds, bottoms)):
        river_data[0].append([0, r, c, stage, cond, bottom])
    
    # Log river statistics
    num_river_cells = len(river_data[0])
    logger.info(f"Generated {num_river_cells} river cells")
    
    if num_river_cells < 10:
        logger.warning(f"Very few river cells ({num_river_cells}) - model may not behave correctly")
    
    # Create visualizations
    from MODGenX.utils_cleanup import plot_diagnostic
    
    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Plot diagnostic maps
    plot_diagnostic(ibound[0], "Active Cells", os.path.join(log_dir, "ibound.png"))
    
    # Create and plot river cells map
    river_cells_map = np.zeros_like(ibound[0])
    for cell in river_data[0]:
        river_cells_map[cell[1], cell[2]] = 1
    plot_diagnostic(river_cells_map, "River Cells", os.path.join(log_dir, "river_cells.png"))
    
    # Plot additional diagnostics
    plot_diagnostic(top, "Top Elevation", os.path.join(log_dir, "top.png"))
    plot_diagnostic(swat_river, "SWAT River", os.path.join(log_dir, "swat_river.png"))
    
    return river_data

def river_correction(swat_river_raster_path, load_raster_args, basin, active):
    """
    Process river raster to prepare it for MODFLOW
    
    Parameters:
    -----------
    swat_river_raster_path : str
        Path to the SWAT river raster
    load_raster_args : dict
        Dictionary containing parameters for raster loading
    basin : numpy.ndarray
        Basin raster array
    active : numpy.ndarray
        Active cells array
        
    Returns:
    --------
    numpy.ndarray
        Processed river raster
    """
    
    # Get fit_to_meter from config if available
    fit_to_meter = 0.3048  # default value
    if 'config' in load_raster_args and hasattr(load_raster_args['config'], 'fit_to_meter'):
        fit_to_meter = load_raster_args['config'].fit_to_meter
        logger.info(f"Using fit_to_meter value from config: {fit_to_meter}")
    
    # Get path_handler if available for log paths
    if 'path_handler' in load_raster_args:
        path_handler = load_raster_args['path_handler']
        log_dir = path_handler.get_log_path("river_correction").rsplit('/', 1)[0]
    else:
        log_dir = "/data/SWATGenXApp/codes/MODFLOW/logs"
    
    with rasterio.open(swat_river_raster_path) as src:
        swat_river = src.read(1)
        nodata = src.nodata
        swat_river = np.where(swat_river == nodata, 0, swat_river)
        swat_river = np.where(swat_river > 0, 1, 0)
        swat_river = swat_river.astype(np.int32)

    assert not np.all(swat_river == 0), "No river cells found in the SWAT river raster"

    return swat_river