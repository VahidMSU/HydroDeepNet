import numpy as np
import flopy
import pandas as pd
import geopandas as gpd
try:
    from MODGenX.utils import *
except ImportError:
    from utils import *
      
from MODGenX.Logger import Logger
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pyproj
import os

logger = Logger(verbose=True)

def create_shapefile_from_modflow_grid_arcpy(BASE_PATH, model_path, MODEL_NAME, out_shp, raster_path):
    """Create a GeoJSON file from MODFLOW grid using the correct raster reference."""
    logger.info(f"Creating grid shapefile from model: {model_path}/{MODEL_NAME}")
    
    # Validate inputs
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Reference raster not found: {raster_path}")
    
    # Get raster extent using GDAL
    from osgeo import gdal
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"Could not open raster file: {raster_path}")
        
    gt = ds.GetGeoTransform()
    x_min_raster = gt[0]
    y_max_raster = gt[3]
    spatial_ref = ds.GetProjection()
    ds = None  # Close dataset
    
    # Load MODFLOW model
    model_nam_file = os.path.join(model_path, f"{MODEL_NAME}.nam")
    if not os.path.exists(model_nam_file):
        raise FileNotFoundError(f"Model NAM file not found: {model_nam_file}")
    
    mf = flopy.modflow.Modflow.load(f"{MODEL_NAME}.nam", model_ws=model_path)
    
    # Get model grid information
    sr = mf.modelgrid
    delr, delc = sr.delr, sr.delc
    angrot = sr.angrot
    epsg_code = 26990  # NAD83 / Illinois East
    
    # Set grid origin to match raster
    xoff, yoff = x_min_raster, y_max_raster
    
    # Compute grid edges
    xedges = np.hstack(([xoff], xoff + np.cumsum(delr)))
    yedges = np.hstack(([yoff], yoff - np.cumsum(delc)))  # Subtract for y-axis convention
    
    # Generate grid cell vertices efficiently
    xedges, yedges = np.meshgrid(xedges, yedges)
    
    # Create arrays for all cell corners at once
    nrow, ncol = sr.nrow, sr.ncol
    bottom_left = list(zip(xedges[:-1, :-1].ravel(), yedges[:-1, :-1].ravel()))
    bottom_right = list(zip(xedges[:-1, 1:].ravel(), yedges[:-1, 1:].ravel()))
    top_right = list(zip(xedges[1:, 1:].ravel(), yedges[1:, 1:].ravel()))
    top_left = list(zip(xedges[1:, :-1].ravel(), yedges[1:, :-1].ravel()))
    
    # Create polygons for each cell
    vertices = [list(box) for box in zip(bottom_left, bottom_right, top_right, top_left, bottom_left)]
    geoms = [Polygon(verts) for verts in vertices]
    
    # Create row and column indices
    rows, cols = np.indices((nrow, ncol))
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'Row': rows.ravel(), 'Col': cols.ravel(), 'geometry': geoms},
        crs=pyproj.CRS.from_epsg(epsg_code)
    )
    
    # Save to files
    gdf.to_file(out_shp)
    gdf.to_file(f'{out_shp}.geojson', driver='GeoJSON')
    
    logger.info(f"Shapefile saved to {out_shp} and GeoJSON to {out_shp}.geojson")
    
    return gdf

def well_location(df_sim_obs, active, NAME, LEVEL, RESOLUTION, load_raster_args):
    """Create an array of well locations for visualization."""
    if df_sim_obs is None or len(df_sim_obs) == 0:
        logger.warning("No observation data provided for well location")
        return np.zeros_like(active[0], dtype=float)
    
    # Initialize array with zeros
    obs_array = np.zeros_like(active[0], dtype=float)
    
    # Extract only needed columns to numpy for faster processing
    obs_np = df_sim_obs[['row', 'col', 'obs_SWL_m']].to_numpy()
    logger.info(f'Number of observations in dataset: {len(obs_np)}')
    
    # Validate array bounds before assignment
    valid_rows = np.where(
        (obs_np[:, 0] >= 0) & 
        (obs_np[:, 0] < obs_array.shape[0]) & 
        (obs_np[:, 1] >= 0) & 
        (obs_np[:, 1] < obs_array.shape[1])
    )[0]
    
    # Assign values using validated indices
    row_indices = obs_np[valid_rows, 0].astype(int)
    col_indices = obs_np[valid_rows, 1].astype(int)
    values = obs_np[valid_rows, 2]
    
    obs_array[row_indices, col_indices] = values
    
    # Generate the final array
    return np.where((active[0] != 0) & (obs_array > 0), 1, 0)

def create_obs_data(row, top, mf, fitToMeter=0.3048):
    """Create a head observation object for MODFLOW."""
    try:
        layer = int(row['layer'])
        r = int(row['row'])
        c = int(row['col'])
        well_id = str(row['wellid'])
        
        # Validate indices are within bounds
        if not (0 <= r < top.shape[0] and 0 <= c < top.shape[1]):
            logger.warning(f"Well {well_id} indices out of bounds: row={r}, col={c}")
            return None
            
        # Calculate head value
        head_value = top[r, c] - fitToMeter * row['SWL']
        time_series_data = [[1, head_value]]
        
        return flopy.modflow.HeadObservation(
            model=mf, 
            layer=layer,
            row=r, 
            column=c, 
            time_series_data=time_series_data, 
            obsname=well_id
        )
    except Exception as e:
        logger.error(f"Error creating observation data: {e}")
        return None

def well_data_import(mf, top, load_raster_args, z_botm, active, grids_path, MODEL_NAME, gpm_to_cmd=5.678, fitToMeter=0.3048):
    """Import and process well data for MODFLOW model."""
    logger.info(f"Importing well data for model: {MODEL_NAME}")
    
    # First, validate that top array doesn't contain NaN values
    if np.isnan(top).any():
        logger.warning(f"NaN values found in top array - fixing...")
        # Get the mean of non-NaN values
        top_mean = np.nanmean(top)
        # Replace NaN with the mean value
        top = np.where(np.isnan(top), top_mean, top)
        logger.info(f"Replaced NaN values in top array with mean value: {top_mean}")
    
    # Also check z_botm arrays for NaNs
    for i, botm in enumerate(z_botm):
        if np.isnan(botm).any():
            logger.warning(f"NaN values found in z_botm layer {i} - fixing...")
            # Get the mean of non-NaN values
            botm_mean = np.nanmean(botm)
            # Replace NaN with the mean value
            z_botm[i] = np.where(np.isnan(botm), botm_mean, botm)
            logger.info(f"Replaced NaN values in z_botm layer {i} with mean value: {botm_mean}")
    
    # Validate input files exist
    observations_path =  "/data/SWATGenXApp/GenXAppData/observations/observations_original.geojson"
    assert os.path.exists(observations_path), f"Observations file not found: {observations_path}"
    assert os.path.exists(grids_path), f"Grids file not found: {grids_path}"

    logger.info(f"Loading observations from {observations_path}")
    obs = gpd.read_file(observations_path)
    
    # Standardize column names
    if "WELLID" in obs.columns:
        obs.rename(columns={"WELLID": "wellid"}, inplace=True)
    
    # Check required columns
    required_cols = ['wellid', 'SWL', 'ELEV_DEM', 'PMP_CPCITY', 'WEL_STATUS', 'AQ_TYPE', 'geometry']
    missing_cols = [col for col in required_cols if col not in obs.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns in observation data: {missing_cols}")
        return None, None, pd.DataFrame()

    logger.info(f"Loading MODFLOW grid from {grids_path}")
    # Load grid data
    grids = gpd.read_file(grids_path)
    # Log CRS information
    logger.info(f"Observation CRS: {obs.crs}")
    logger.info(f"Grid CRS: {grids.crs}")
    
    # Ensure consistent column naming
    if 'Row' in grids.columns:
        grids = grids.rename(columns={'Row': 'row'})
        if 'Row' in grids.columns:
            grids.drop(columns=['Row'], inplace=True, errors='ignore')
    
    if 'Col' in grids.columns:
        grids = grids.rename(columns={'Col': 'col'})
        if 'Col' in grids.columns:
            grids.drop(columns=['Col'], inplace=True, errors='ignore')
    
    # Ensure we have row and col in grids
    if 'row' not in grids.columns or 'col' not in grids.columns:
        logger.error("Grid data missing required row/col columns")
        return None, None, pd.DataFrame()
    
    # Ensure CRS match
    logger.info(f"Reprojecting observations to match grid CRS: {grids.crs}")
    obs = obs.to_crs(grids.crs)

    # Remove existing row/col columns from observations if present
    # This prevents conflicts with the spatial join
    if 'col' in obs.columns and 'row' in obs.columns:
        obs = obs.drop(columns=['col', 'row'])

    # Get model grid dimensions
    nrow, ncol = top.shape
    logger.info(f"Model grid dimensions: {nrow} rows x {ncol} columns")
    
    # Perform spatial join
    logger.info("Performing spatial join between observations and grid")
    df_obs = gpd.sjoin_nearest(obs, grids, how='inner', max_distance=250)
    
    # Check if the join worked
    if len(df_obs) == 0:
        logger.warning("No observations found in grid - trying with buffered points")
        # Try with buffer around points
        buffer_dist = min(mf.modelgrid.delr.min(), mf.modelgrid.delc.min()) / 2
        obs_buffered = obs.copy()
        obs_buffered['geometry'] = obs_buffered.geometry.buffer(buffer_dist)
        df_obs = gpd.sjoin(obs_buffered, grids, how='inner')
    
    # Check if we have observations
    if len(df_obs) == 0:
        logger.error("No observations found within model grid after spatial operations")
        return None, None, pd.DataFrame()
    
    logger.info(f"Joined observations before: {len(df_obs)} records")
    
    # Log column names to help diagnose issues
    logger.info(f"Columns after spatial join: {df_obs.columns.tolist()}")
    
    # Make sure we have row/col columns
    # If we don't have them, they might be under different names
    if 'row' not in df_obs.columns or 'col' not in df_obs.columns:
        # Find row/col columns in all variations (row, Row, row_left, etc.)
        row_cols = [col for col in df_obs.columns if 'row' in col.lower()]
        col_cols = [col for col in df_obs.columns if 'col' in col.lower()]
        
        if row_cols:
            df_obs['row'] = df_obs[row_cols[0]]
            logger.info(f"Using column {row_cols[0]} for row values")
        else:
            logger.error("Cannot find row column in joined data")
            return None, None, pd.DataFrame()
            
        if col_cols:
            df_obs['col'] = df_obs[col_cols[0]]
            logger.info(f"Using column {col_cols[0]} for col values")
        else:
            logger.error("Cannot find col column in joined data")
            return None, None, pd.DataFrame()
    
    # Convert numeric columns to ensure proper comparison
    for col in ['SWL', 'ELEV_DEM', 'PMP_CPCITY', 'row', 'col']:
        if col in df_obs.columns:
            df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
    
    # Clean data - drop NaN values in critical columns
    df_obs = df_obs.dropna(subset=['SWL', 'ELEV_DEM', 'PMP_CPCITY', 'row', 'col'])
    
    # Make sure values are reasonable
    # Filter out negative or zero values where inappropriate
    df_obs = df_obs[df_obs['SWL'] > 0]
    df_obs = df_obs[df_obs['ELEV_DEM'] > 0]
    
    # Limit pumping rates to reasonable values
    MAX_PUMP_RATE = 10000  # maximum reasonable pump rate in gpm
    df_obs['PMP_CPCITY'] = df_obs['PMP_CPCITY'].clip(0, MAX_PUMP_RATE)
    
    # Log statistics 
    logger.info(f"Observations after filtering: {len(df_obs)}")
    
    # Check if we have any valid wells left
    if len(df_obs) == 0:
        logger.warning("No valid observations remaining after filtering")
        return None, None, pd.DataFrame()
    
    logger.info(f"SWL range: {df_obs.SWL.min()} to {df_obs.SWL.max()}")
    logger.info(f"ELEV_DEM range: {df_obs.ELEV_DEM.min()} to {df_obs.ELEV_DEM.max()}")
    
    # Convert row/col to integers and ensure they're within model bounds
    df_obs['row'] = df_obs['row'].astype(int)
    df_obs['col'] = df_obs['col'].astype(int)
    
    # Filter out wells outside model boundaries
    valid_wells = (
        (df_obs['row'] >= 0) & 
        (df_obs['row'] < nrow) & 
        (df_obs['col'] >= 0) & 
        (df_obs['col'] < ncol)
    )
    df_obs = df_obs[valid_wells]
    
    if len(df_obs) == 0:
        logger.warning("No wells within valid model grid boundaries")
        return None, None, pd.DataFrame()
    
    logger.info(f"Wells within valid grid bounds: {len(df_obs)}")
    
    # Save diagnostic plot
    plt.figure(figsize=(10, 8))
    df_obs.plot(column='SWL', legend=True)
    plt.title('Well Static Water Levels')
    log_dir = '/data/SWATGenXApp/codes/MODFLOW/logs'
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f'{log_dir}/{MODEL_NAME}_obs_heads.png')
    plt.close()
    
    # Process observation data
    logger.info("Processing observation data")
    
    # Extract row and column arrays
    rows = df_obs['row'].values
    cols = df_obs['col'].values
    
    logger.info(f"Rows: {rows.min()} to {rows.max()}")  
    logger.info(f"Cols: {cols.min()} to {cols.max()}")
    
    # Calculate heads using vectorized operations
    heads = fitToMeter * (df_obs['ELEV_DEM'].values - df_obs['SWL'].values)
    
    # Validate heads - make sure they're not NaN and within reasonable range
    invalid_heads = np.isnan(heads)
    if np.any(invalid_heads):
        logger.warning(f"Found {np.sum(invalid_heads)} NaN head values - filtering...")
        valid_head_mask = ~invalid_heads
        heads = heads[valid_head_mask]
        rows = rows[valid_head_mask]
        cols = cols[valid_head_mask]
        df_obs = df_obs[~invalid_heads.tolist()]
    
    if len(df_obs) == 0:
        logger.warning("No valid heads remaining after filtering")
        return None, None, pd.DataFrame()
    
    # Assign wells to appropriate model layers
    # Pre-allocate array
    z_botm_well_loc = np.zeros((len(z_botm), len(rows)))
    
    # Fill the array
    for i in range(len(z_botm)):
        for j, (r, c) in enumerate(zip(rows, cols)):
            # Ensure model array values are not NaN
            botm_val = z_botm[i][r, c]
            if np.isnan(botm_val):
                # Get mean of surrounding cells
                r_min = max(0, r-1)
                r_max = min(nrow-1, r+1)
                c_min = max(0, c-1)
                c_max = min(ncol-1, c+1)
                neighbor_vals = z_botm[i][r_min:r_max+1, c_min:c_max+1]
                botm_val = np.nanmean(neighbor_vals)
                if np.isnan(botm_val):
                    # Still NaN, use layer average
                    botm_val = np.nanmean(z_botm[i])
                z_botm[i][r, c] = botm_val
            
            z_botm_well_loc[i, j] = botm_val
    
    # Find layer for each well
    heads_below_layer = heads[None, :] <= z_botm_well_loc
    
    # Find first layer where head is below layer bottom
    layer = np.argmax(heads_below_layer, axis=0)
    
    # Handle case where head is not below any layer bottom (assign to top active layer)
    mask_no_layer = ~np.any(heads_below_layer, axis=0)
    layer[mask_no_layer] = 0
    
    # Assign layers to DataFrame
    df_obs['layer'] = layer
    
    # Create pumping data for wells
    pumping_rates = -1 * gpm_to_cmd * df_obs['PMP_CPCITY'].values
    
    # Validate pumping rates (ensure no NaN values)
    if np.isnan(pumping_rates).any():
        logger.warning("NaN pumping rates found - setting to minimal value")
        pumping_rates = np.where(np.isnan(pumping_rates), -0.1, pumping_rates)
    
    # Create well data tuples
    well_data = []
    for i, (lay, r, c, q) in enumerate(zip(layer, rows, cols, pumping_rates)):
        # Final validation of indices
        if 0 <= lay < len(z_botm)+1 and 0 <= r < nrow and 0 <= c < ncol:
            well_data.append((lay, r, c, q))
    
    # Update the dataframe
    df_obs = df_obs.iloc[:len(well_data)].copy()  # Trim to match well_data length
    df_obs['wel_data'] = well_data
    
    # Create well data for MODFLOW (stress periods)
    wel_data = {per: well_data for per in range(mf.nper)}
    
    # Create observation data
    obs_data = []
    for _, row in df_obs.iterrows():
        obs = create_obs_data(row, top, mf, fitToMeter)
        if obs is not None:
            obs_data.append(obs)
    
    if not obs_data:
        logger.warning("No valid observation data created")
        return wel_data, None, df_obs
    
    logger.info(f"Created {len(obs_data)} observation points")
    
    return wel_data, obs_data, df_obs

