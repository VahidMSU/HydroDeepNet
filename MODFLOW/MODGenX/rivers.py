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
    print(f"river_gen: nrow={nrow}, ncol={ncol}")
    print(f"river_gen: swat_river.shape={swat_river.shape}, top.shape={top.shape}, ibound.shape={ibound.shape}")
    

    # Print range of values for key arrays
    print(f"swat_river values range: {np.min(swat_river)} to {np.max(swat_river)}")
    print(f"top values range: {np.min(top)} to {np.max(top)}")
    
    river_data = {0: []}  # Initialize an empty list for layer 0
    
   
    
    for i in range(nrow):
        for j in range(ncol):
            # Check conditions to identify river cells
            if swat_river[i,j] > 0 and top[i,j] > 0:
                # Append river cell data to river_data
                river_data[0].append([0, i, j, top[i,j] + 1, swat_river[i,j], top[i,j] - 1])
    
    assert len(river_data[0]) > 10, "No river cells found"

    # Create visualizations
    # Plot ibound
    plt.close()
    plt.imshow(ibound[0])   
    plt.colorbar()
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/MODGenX/ibound.png')
    plt.close()
    
    # Plot river cells
    river_cells_map = np.zeros_like(ibound[0])
    for cell in river_data[0]:
        river_cells_map[cell[1], cell[2]] = 1
    plt.imshow(river_cells_map)
    plt.colorbar()
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/MODGenX/river_cells.png')
    plt.close()
    
    # Plot top elevation
    plt.imshow(top)
    plt.colorbar()
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/MODGenX/top.png')
    plt.close()
    
    # Plot swat_river
    plt.imshow(swat_river)
    plt.colorbar()
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/MODGenX/swat_river.png')
    plt.close()
    
    time.sleep(5)
    logger.info(f'Number of river cells: {len(river_data[0])}')
    return river_data

def river_correction(swat_river_raster_path, load_raster_args, basin, active):
    """Process river raster to prepare it for MODFLOW"""
    result = load_raster(swat_river_raster_path, load_raster_args)
    print(f"River raster loaded - min: {np.min(result)}, max: {np.max(result)}, mean: {np.mean(result)}")
    
    # Check for infinite values
    inf_count = np.sum(np.isinf(result))
    if inf_count > 0:
        print(f"Found {inf_count} infinite values in river raster")
    
    # Replace infinite values with zeros
    result = np.where(np.isinf(result), 0, result)
    
    # Match dimensions with basin
    result = match_raster_dimensions(basin, result)
    
    # Count river cells before applying active mask
    river_cells_before = np.sum(result > 0)
    print(f"River cells before masking: {river_cells_before}")
    
    # Don't mask river cells by active domain - this prevents river cells from being removed
    # when they should be included in the model
    # result = np.where(active[0] == 0, 0, result)
    
    # Instead, just ensure any NoData values (9999) are properly handled
    if np.any(result == 9999):
        print("Warning: Found possible NoData values (9999) in river raster")
        result = np.where(result == 9999, 0, result)
    
    # Count river cells after processing
    river_cells_after = np.sum(result > 0)
    print(f"River cells after processing: {river_cells_after}")
    
    return result
