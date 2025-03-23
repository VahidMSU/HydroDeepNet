import numpy as np
import flopy
import pandas as pd
import geopandas as gpd
try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
      
from MODGenX.Logger import Logger

logger = Logger(verbose=True)


def well_location(df_sim_obs, active, NAME, LEVEL, RESOLUTION, load_raster_args):
    # Assuming active is a 3D array, and we are using its first 2D slice for comparison
    # Initialize obs_array with the same shape as the first slice of active
    obs_array = np.zeros_like(active[0], dtype=float)

    obs_np = df_sim_obs[['row', 'col', 'obs_SWL_m']].to_numpy()
    logger.info(f'number of observations in the original dataset: {len(obs_np)}')

    # Fill obs_array with SWL values using numpy array
    for row in obs_np:
        row_idx, col_idx = int(row[0]), int(row[1])
        if 0 <= row_idx < obs_array.shape[0] and 0 <= col_idx < obs_array.shape[1]:
            obs_array[row_idx, col_idx] = row[2]
        else:
            logger.info(f"Skipping out-of-bounds index: row={row_idx}, col={col_idx}")

    return np.where((active[0] != 0) & (obs_array > 0), 1, 0)


def create_obs_data(row,top, mf, fitToMeter=0.3048, obs_index=0):
    time_series_data = [[1, top[int(row['row']), int(row['col'])] - fitToMeter * row['SWL']]]
    return flopy.modflow.HeadObservation(
        model=mf, 
        layer=int(row['layer']),  # changed to dynamic layering based on the DataFrame
        row=int(row['row']), 
        column=int(row['col']), 
        time_series_data=time_series_data, 
        obsname = str(row['wellid'])  # changed to dynamic obsname based on index
    )


def well_data_import(mf,top, load_raster_args, z_botm, active, grids_path, MODEL_NAME, gpm_to_cmd=5.678, fitToMeter=0.3048):
    """ 
    path_to_top_raster - Path to the top raster file, used in the function get_row_col_from_coords.
    z_botm - Array containing the bottom elevations of the model layers.
    active - Array inDICating active and inactive cells in the model grid.
    mf - Represents the MODFLOW model object, used to get the number of periods (mf.nper)."""
    ref_raster_path = load_raster_args['active']
    LEVEL = load_raster_args['LEVEL']
    NAME = load_raster_args['NAME']
    RESOLUTION = load_raster_args['RESOLUTION']
    username = load_raster_args['username']
    VPUID = load_raster_args['VPUID']
    database_paths = database_file_paths()

    assert os.path.exists(database_paths['observations']), f"Observations file not found: {database_paths['observations']}"
    assert os.path.exists(grids_path), f"Grids file not found: {grids_path}"

    logger.info(f"Loading observations from {database_paths['observations']}")
    logger.info(f"Loading MODFLOW grid from {grids_path}")

    # Load the observation data
    logger.info(f"Loading observations from {database_paths['observations']}")
    obs = gpd.read_file(database_paths['observations'])
    if "WELLID" in obs.columns:
        obs.rename(columns={"WELLID": "wellid"}, inplace=True)
        
    # Check if the required columns exist
    required_cols = ['wellid', 'SWL', 'ELEV_DEM', 'PMP_CPCITY', 'WEL_STATUS', 'AQ_TYPE', 'geometry']

    for col in required_cols:
         assert col in obs.columns, f"Missing required column: {col}"

    obs = obs[required_cols]    

    # Load the MODFLOW grid
    logger.info(f"Loading MODFLOW grid from {grids_path}")
    grids = gpd.read_file(grids_path)
    
    ## plot grid
    grids.plot()
    plt.title('Grid')
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/logs/grid.png')
    plt.close()

    ## make sure both are the same crs
    obs = obs.to_crs(grids.crs)
    
    # Check column names
    logger.info(f"Grid columns: {grids.columns}")
    
    # Handle different column naming conventions
    if 'Row' in grids.columns:
        grids['row'] = grids['Row']
    if 'Col' in grids.columns:
        grids['col'] = grids['Col']
        
    assert 'row' in grids.columns, "Missing required column: row"   
    assert 'col' in grids.columns, "Missing required column: col"


    # Perform spatial join
    logger.info("Performing spatial join between observations and grid")
    
    assert len(obs) > 0, "No observations found"
    assert len(grids) > 0, "No grid found"

    df_obs = obs.sjoin(grids, how='inner')

    logger.info(f"Joined observations: {len(df_obs)} records")
    
    assert len(df_obs) > 0, "No observations found in the grid"
    # Create unique well IDs if they don't exist
    if 'wellid' not in df_obs.columns:
        df_obs['wellid'] = np.arange(0, len(df_obs))
    
    # Filter observations to valid ones
    filter_conditions = (
        df_obs.SWL.notna() & 
        df_obs.ELEV_DEM.notna()
    #    (df_obs.SWL <= 999) & 
     #   df_obs.PMP_CPCITY.notna() & 
     #   (df_obs.PMP_CPCITY > 10) & 
     #   (df_obs.WEL_STATUS == 'ACT') &
     #   (df_obs.AQ_TYPE != 'ROCK')
    )

    ### plot obs heads
    import matplotlib.pyplot as plt
    df_obs['SWL'] = df_obs['SWL'].astype(float)
    df_obs.plot(column='SWL', legend=True)
    plt.title('Observed Heads')
    plt.savefig('/data/SWATGenXApp/codes/MODFLOW/logs/obs_heads.png')
    plt.close()

    ### some loggins
    logger.info(f"Range OF SWL: {df_obs.SWL.min()} to {df_obs.SWL.max()}")
    logger.info(f"Range OF ELEV_DEM: {df_obs.ELEV_DEM.min()} to {df_obs.ELEV_DEM.max()}")
    logger.info(f"Range OF PMP_CPCITY: {df_obs.PMP_CPCITY.min()} to {df_obs.PMP_CPCITY.max()}")
    logger.info(f"Unique values of WEL_STATUS: {df_obs.WEL_STATUS.unique()}")
    logger.info(f"Unique values of AQ_TYPE: {df_obs.AQ_TYPE.unique()}")
    logger.info(f"Unique values of wellid: {df_obs.wellid.unique()}")
    logger.info(f"Unique values of row: {df_obs.row.unique()}")
    logger.info(f"Unique values of col: {df_obs.col.unique()}")

    df_obs = df_obs[filter_conditions]

    assert len(df_obs) > 0, "No valid observations found"
    
    logger.info(f"Filtered observations: {len(df_obs)} records")
    
    # If no observations remain after filtering, return empty data
    if len(df_obs) == 0:
        logger.warning("No valid observations remaining after filtering")
        return None, None, None
    
    # Convert row and col to integers
    rows = df_obs['row'].values.astype(int)
    cols = df_obs['col'].values.astype(int)
    
    # Calculate heads
    heads = fitToMeter * (df_obs['ELEV_DEM'].values - df_obs['SWL'].values)
    
    # Get the bottom elevations at the well locations
    z_botm_well_loc = np.array([z_botm[i][rows, cols] for i in range(len(z_botm))])
    
    # Identify the layer where the SWL is above the bottom elevation of the layer
    layer = np.argmax((heads[None, :] <= z_botm_well_loc), axis=0)
    
    # If SWL is below all layers, assign to the last active layer
    layer[np.all(heads[None, :] > z_botm_well_loc, axis=0)] = len(z_botm) - 2
    
    # Assign layers to DataFrame
    df_obs['layer'] = layer
    
    # Create well data
    df_obs['wel_data'] = list(zip(df_obs['layer'], rows, cols, -1 * gpm_to_cmd * df_obs['PMP_CPCITY']))
    
    # Check if wells are in active cells
    active_status = np.array([active[l][r, c] for l, r, c in zip(layer, rows, cols)])
    df_obs['is_active'] = ~np.isin(active_status, [-1, 2, 0])
    
    # Filter to active wells
    df_obs = df_obs[df_obs['is_active']]
    
    # Create well data for MODFLOW
    wel_data = {per: list(df_obs['wel_data']) for per in range(mf.nper)}
    
    # Convert to integers for observation data
    df_obs[['layer', 'row', 'col', 'wellid']] = df_obs[['layer', 'row', 'col', 'wellid']].astype(int)
    
    # Final check for data
    if len(df_obs) == 0:
        logger.warning("No active wells found")
        return None, None, None
    
    # Create observation data
    obs_data = df_obs[['layer', 'wellid', 'row', 'col', 'SWL']].apply(
        lambda row: create_obs_data(row, top, mf), axis=1
    ).tolist()
    
    return wel_data, obs_data, df_obs
    
