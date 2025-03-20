
try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
import numpy as np

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
    
    river_data = {0: []}  # Initialize an empty list for layer 0
    
    for i in range(nrow):
        for j in range(ncol):
            # Check conditions to identify river cells
            if swat_river[i,j] != 0 and top[i,j] > 0 and ibound[0,i,j] == 1:
                # Append river cell data to river_data
                river_data[0].append([0, i, j, top[i,j] + 1, swat_river[i,j], top[i,j] - 1])
    print(f'############### length of river data {len(river_data[0])}  ########################')            
    
    return river_data

def river_correction(swat_river_raster_path, load_raster_args, basin, active):
    result = load_raster(swat_river_raster_path, load_raster_args)

    result = np.where(np.isinf(result) == True, 0, result)

    result = match_raster_dimensions(basin, result)

    result = np.where(active[0] == 0, 0, result)

    return result
