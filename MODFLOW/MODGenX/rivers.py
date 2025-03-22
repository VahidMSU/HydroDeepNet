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
    

    # Print range of values for key arrays
    logger.info(f"swat_river values range: {np.min(swat_river)} to {np.max(swat_river)}")
    logger.info(f"top values range: {np.min(top)} to {np.max(top)}")
    
    river_data = {0: []}  # Initialize an empty list for layer 0
    
    assert top.shape == swat_river.shape, "top and swat_river arrays must have the same shape"

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
    
    with rasterio.open(swat_river_raster_path) as src:
        swat_river = src.read(1)
        nodata = src.nodata
        swat_river = np.where(swat_river == nodata, 0, swat_river)
        swat_river = np.where(swat_river > 0, 1, 0)
        swat_river = swat_river.astype(np.int32)

    assert not np.all(swat_river == 0), "No river cells found in the SWAT river raster"


    return swat_river