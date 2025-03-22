try:
    from MODGenX.utils import *
except ImportError:
    from utils import *
import numpy as np

from MODGenX.Logger import Logger


logger = Logger(verbose=True)

def river_gen(nrow, ncol, swat_river, top, ibound):
    """
    Generate river data based on given conditions.
    
    Parameters:
    - nrow: Number of rows in the grid
    - ncol: Number of columns in the grid
    - swat_river: 2D array containing SWAT river data
    - top: 2D array containing top elevation data
    - ibound: 3D array containing cell activity data
    
    Returns:
    - river_data: Dictionary containing information about river cells
    """
    logger.info(f"river_gen: nrow={nrow}, ncol={ncol}")
    logger.info(f"river_gen: swat_river.shape={swat_river.shape}, top.shape={top.shape}, ibound.shape={ibound.shape}")
    
    # Add some diagnostics about valid cells
    active_cells = np.sum(ibound[0] == 1)
    boundary_cells = np.sum(ibound[0] == -1)
    inactive_cells = np.sum(ibound[0] == 0)
    total_cells = ibound[0].size
    
    logger.info(f"Active cells: {active_cells} ({100*active_cells/total_cells:.2f}%)")
    logger.info(f"Boundary cells: {boundary_cells} ({100*boundary_cells/total_cells:.2f}%)")
    logger.info(f"Inactive cells: {inactive_cells} ({100*inactive_cells/total_cells:.2f}%)")
    
    # Detailed diagnostics on top values
    top_min, top_max = np.min(top), np.max(top)
    top_mean = np.mean(top)
    top_unique = np.unique(top)
    
    logger.info(f"swat_river values range: {np.min(swat_river)} to {np.max(swat_river)}")
    logger.info(f"top values range: {top_min} to {top_max} (mean: {top_mean:.2f})")
    assert top_min != top_max, "Top elevation data is constant"
    if len(top_unique) < 10:
        logger.warning(f"Only {len(top_unique)} unique values in top elevation data: {top_unique}")
    
    if top_min > 900000 or top_max > 900000:
        logger.error("ERROR: Top elevation values are extreme (>900000). This indicates a data loading issue.")
        logger.error("Check the source DEM raster for additional bands or other data issues.")
        raise ValueError("Invalid top elevation data - extreme values detected")
    
    river_data = {0: []}  # Initialize an empty list for layer 0
    
    # Count potential river cells
    river_cells = np.sum(swat_river > 0)
    river_and_active = np.sum((swat_river > 0) & (ibound[0] == 1))
    logger.info(f"Potential river cells: {river_cells}")
    logger.info(f"River cells in active domain: {river_and_active}")
    
    for i in range(nrow):
        for j in range(ncol):
            # Check conditions to identify river cells
            if swat_river[i,j] > 0 and top[i,j] > 0 and ibound[0,i,j] == 1:
                # Append river cell data to river_data
                river_data[0].append([0, i, j, top[i,j] + 1, swat_river[i,j], top[i,j] - 1])
    
    # If no river cells were found, create at least one fallback cell in an active area
    if len(river_data[0]) == 0:
        logger.info("Warning: No river cells found. Creating fallback river cells.")
        # Find locations where there are active cells
        active_locs = np.where(ibound[0] == 1)
        if len(active_locs[0]) > 0:
            # Select a few points from active cells
            max_points = min(5, len(active_locs[0]))
            step = max(1, len(active_locs[0]) // max_points)
            
            for idx in range(0, len(active_locs[0]), step)[:max_points]:
                i, j = active_locs[0][idx], active_locs[1][idx]
                conductance = 100.0  # Default conductance value
                river_data[0].append([0, i, j, top[i,j] + 1, conductance, top[i,j] - 1])
                logger.info(f"Added fallback river cell at row={i}, col={j}")
    
    logger.info(f'Number of river cells: {len(river_data[0])}')
    return river_data

def river_correction(swat_river_raster_path, load_raster_args, basin, active):
    """Process river raster to prepare it for MODFLOW"""
    result = load_raster(swat_river_raster_path, load_raster_args)
    logger.info(f"River raster loaded - min: {np.min(result)}, max: {np.max(result)}, mean: {np.mean(result)}")
    
    # Check for infinite values
    inf_count = np.sum(np.isinf(result))
    if inf_count > 0:
        logger.info(f"Found {inf_count} infinite values in river raster")
    
    # Replace infinite values with zeros
    result = np.where(np.isinf(result), 0, result)
    
    # Match dimensions with basin
    result = match_raster_dimensions(basin, result)
    
    # Count river cells before applying active mask
    river_cells_before = np.sum(result > 0)
    logger.info(f"River cells before masking: {river_cells_before}")
    
    # Don't mask river cells by active domain - this prevents river cells from being removed
    # when they should be included in the model
    # result = np.where(active[0] == 0, 0, result)
    
    # Instead, just ensure any NoData values (-999) are properly handled
    if np.any(result == -999):
        logger.info("Warning: Found possible NoData values (-999) in river raster")
        result = np.where(result == -999, 0, result)
    
    # Count river cells after processing
    river_cells_after = np.sum(result > 0)
    logger.info(f"River cells after processing: {river_cells_after}")
    
    return result
