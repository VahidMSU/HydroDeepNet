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
from typing import Dict, List, Tuple, Optional, Union

logger = Logger(verbose=True)

def create_shapefile_from_modflow_grid_arcpy(
    BASE_PATH: str, 
    model_path: str, 
    MODEL_NAME: str, 
    out_shp: str, 
    raster_path: str
) -> gpd.GeoDataFrame:
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

def well_location(
    df_sim_obs: pd.DataFrame, 
    active: np.ndarray, 
    NAME: str, 
    LEVEL: str, 
    RESOLUTION: Union[int, str], 
    load_raster_args: Dict
) -> np.ndarray:
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

def load_observations(observations_path, verbose=False):
    """
    Load and preprocess observation data.
    """
    if not os.path.exists(observations_path):
        logger.error(f"Observations file not found: {observations_path}")
        
        # Try alternative observation file if original one is not found
        alt_path = observations_path.replace('observations_original.geojson', 'observations.geojson')
        if os.path.exists(alt_path):
            logger.info(f"Using alternative observations file: {alt_path}")
            observations_path = alt_path
        else:
            raise FileNotFoundError(f"Observations file not found: {observations_path}")
    
    try:
        logger.info(f"Loading observations from {observations_path}")
        obs = gpd.read_file(observations_path)
        
        # Log observation details for debugging
        logger.info(f"loaded obs columns: {obs.columns.tolist()}")
        logger.info(f"loaded obs records: {len(obs)}")
        logger.info(f"loaded obs crs: {obs.crs}")
        
        # Immediately reset index to avoid duplicate index issues
        obs = obs.reset_index(drop=True)

        # Handle the different possible column structures
        # Check if this is the simplified observations.geojson (fewer columns)
        if 'ELEV_DEM' not in obs.columns and 'PMP_CPCITY' not in obs.columns:
            logger.info("Using simplified observations file format")
            
            # Add required columns with default values if missing
            if 'WEL_STATUS' not in obs.columns:
                obs['WEL_STATUS'] = 'Active'
            
            if 'ELEV_DEM' not in obs.columns and 'TOWNSHIP' in obs.columns:
                # Use a fixed elevation value if ELEV_DEM is missing
                obs['ELEV_DEM'] = 300.0
                logger.info("Created default ELEV_DEM column with value 300.0")
            
            if 'PMP_CPCITY' not in obs.columns:
                obs['PMP_CPCITY'] = 0.0
                logger.info("Created default PMP_CPCITY column with value 0.0")
            
            if 'AQ_TYPE' not in obs.columns:
                obs['AQ_TYPE'] = 'Unknown'
        
        # Standardize common column names
        col_mapping = {
            "WELLID": "wellid",
            "WELL_ID": "wellid",
            "Well_ID": "wellid",
            "STATIC_LEVEL": "SWL",
            "SWL_FT": "SWL",
            "STATIC_WATER_LEVEL": "SWL",
            "ELEVATION": "ELEV_DEM",
            "ELEV": "ELEV_DEM",
            "DEM": "ELEV_DEM",
            "PUMP_CAPACITY": "PMP_CPCITY",
            "PUMP_RATE": "PMP_CPCITY",
            "PUMPING_CAPACITY": "PMP_CPCITY",
            "STATUS": "WEL_STATUS",
            "WELL_STATUS": "WEL_STATUS",
            "AQUIFER": "AQ_TYPE",
            "AQUIFER_TYPE": "AQ_TYPE",
            "Row": "row",
            "Col": "col",
        }
        
        # Apply column renaming for columns that exist
        cols_to_rename = {old: new for old, new in col_mapping.items() 
                         if old in obs.columns and new not in obs.columns}
        if cols_to_rename:
            obs = obs.rename(columns=cols_to_rename)
            if verbose:
                logger.info(f"Renamed columns: {cols_to_rename}")
        
        # Ensure all required columns exist
        required_cols = ['wellid', 'SWL', 'ELEV_DEM', 'PMP_CPCITY', 'WEL_STATUS', 'AQ_TYPE', 'geometry']
        
        # Handle missing wellid column specifically
        if 'wellid' not in obs.columns:
            if 'WELLID' in obs.columns:
                obs['wellid'] = obs['WELLID']
                logger.info("Using WELLID column as wellid")
            elif 'OBJECTID' in obs.columns:
                obs['wellid'] = obs['OBJECTID'].astype(str)
                logger.info("Using OBJECTID as wellid")
            else:
                obs['wellid'] = [f"WELL_{i}" for i in range(len(obs))]
                logger.info("Created synthetic wellid values")
        
        # Check for other missing columns and create defaults
        for col in required_cols:
            if col not in obs.columns and col != 'geometry':
                if col == 'SWL':
                    if 'WELL_DEPTH' in obs.columns:
                        obs['SWL'] = obs['WELL_DEPTH'] * 0.3  # Approximate SWL as 30% of well depth
                        logger.info("Approximated SWL from WELL_DEPTH")
                    else:
                        obs['SWL'] = 20.0  # Default value
                        logger.info("Created default SWL column with value 20.0")
                elif col == 'ELEV_DEM':
                    obs['ELEV_DEM'] = 300.0  # Default elevation
                    logger.info("Created default ELEV_DEM column with value 300.0")
                elif col == 'PMP_CPCITY':
                    obs['PMP_CPCITY'] = 0.0
                    logger.info("Created default PMP_CPCITY column with value 0.0")
                elif col == 'WEL_STATUS':
                    obs['WEL_STATUS'] = 'Active'
                    logger.info("Created default WEL_STATUS column with value 'Active'")
                elif col == 'AQ_TYPE':
                    obs['AQ_TYPE'] = 'Unknown'
                    logger.info("Created default AQ_TYPE column with value 'Unknown'")
        
        # Convert columns to numeric safely
        for col in ['SWL', 'ELEV_DEM', 'PMP_CPCITY']:
            if col in obs.columns:
                try:
                    logger.info(f"Converting {col} to numeric")
                    # Force conversion to string first to avoid list/tuple errors
                    obs[col] = pd.to_numeric(obs[col].astype(str), errors='coerce')
                except Exception as e:
                    logger.error(f"Error converting {col} to numeric: {str(e)}")
                    # Set default values for the column
                    if col == 'SWL':
                        obs[col] = 20.0
                    elif col == 'ELEV_DEM':
                        obs[col] = 300.0
                    elif col == 'PMP_CPCITY':
                        obs[col] = 0.0
        
        # Filter data
        logger.info(f"Observation records before filtering: {len(obs)}")
        
        # Drop rows with NaN in critical columns
        obs = obs.dropna(subset=['SWL', 'ELEV_DEM', 'geometry'])
        logger.info(f"Observations after dropping NaN values: {len(obs)}")
        
        # Filter to keep only positive values for numeric columns
        valid_mask = (obs['SWL'] > 0) & (obs['ELEV_DEM'] > 0)
        obs = obs[valid_mask].copy()
        logger.info(f"Observations after filtering non-positive values: {len(obs)}")
        
        # Make sure there are no duplicate indices
        if obs.index.duplicated().any():
            obs = obs.reset_index(drop=True)
            logger.info("Reset index to eliminate duplicate indices")
        
        # Limit pump capacity to reasonable values
        if 'PMP_CPCITY' in obs.columns:
            obs['PMP_CPCITY'] = obs['PMP_CPCITY'].clip(0, 10000)
        
        # Make sure we have reasonable data
        if len(obs) == 0:
            logger.error("No observations left after filtering")
            return None
        
        logger.info(f"Final observation count: {len(obs)}")
        logger.info(f"SWL range: {obs['SWL'].min():.2f} to {obs['SWL'].max():.2f}")
        logger.info(f"ELEV_DEM range: {obs['ELEV_DEM'].min():.2f} to {obs['ELEV_DEM'].max():.2f}")
        
        # Return preprocessed observations
        return obs
    
    except Exception as e:
        logger.error(f"Error loading observations: {str(e)}")
        
        # Try the alternate observations file as a fallback
        alt_path = observations_path.replace('observations_original.geojson', 'observations.geojson')
        if observations_path != alt_path and os.path.exists(alt_path):
            logger.info(f"Trying alternate observations file: {alt_path}")
            try:
                return load_observations(alt_path, verbose)
            except Exception as alt_e:
                logger.error(f"Error loading alternate observations: {str(alt_e)}")
        
        # Return None if all attempts fail
        return None

def well_data_import(mf, top, load_raster_args, z_botm, active, grids_path, MODEL_NAME, gpm_to_cmd=5.678, fitToMeter=0.3048):
    """
    Import and process well data for MODFLOW model.
    """
    logger.info(f"Importing well data for model: {MODEL_NAME}")
    

    path_handler = load_raster_args['path_handler']
    fitToMeter = path_handler.config.fit_to_meter
    log_dir = os.path.dirname(path_handler.get_log_path("well_data"))
    observations_path = path_handler.get_database_file_paths()['observations']

    obs = load_observations(observations_path, verbose=True)

    if obs is None or len(obs) == 0:
        logger.error("No valid observations found in any dataset")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Ensure Grids_MODFLOW file exists
    if not os.path.exists(grids_path):
        logger.error(f"Grids file not found: {grids_path}")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Load grid data
    logger.info(f"Loading MODFLOW grid from {grids_path}")
    try:
        grids = gpd.read_file(grids_path)
    except Exception as e:
        logger.error(f"Error loading grid file: {str(e)}")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Log CRS information
    logger.info(f"Observation CRS: {obs.crs}")
    logger.info(f"Grid CRS: {grids.crs}")
    logger.info(f"Observation columns: {obs.columns.tolist()}")
    logger.info(f"Grid columns: {grids.columns.tolist()}")
    
    # Standardize grid column names
    if 'Row' in grids.columns and 'row' not in grids.columns:
        grids = grids.rename(columns={'Row': 'row'})
        logger.info("Renamed 'Row' to 'row' in grid data")
    
    if 'Col' in grids.columns and 'col' not in grids.columns:
        grids = grids.rename(columns={'Col': 'col'})
        logger.info("Renamed 'Col' to 'col' in grid data")
    
    # Ensure we have row and col in grids
    if 'row' not in grids.columns or 'col' not in grids.columns:
        logger.error(f"Grid data missing required row/col columns. Available: {grids.columns.tolist()}")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Make sure both dataframes have the same CRS
    try:
        if grids.crs != obs.crs:
            logger.info(f"Reprojecting observations to match grid CRS: {grids.crs}")
            obs = obs.to_crs(grids.crs)
    except Exception as e:
        logger.error(f"Error reprojecting observations: {str(e)}")
        # Continue anyway - the spatial join will handle CRS differences
    
    # Ensure there are no duplicate indices
    if obs.index.duplicated().any():
        obs = obs.reset_index(drop=True)
    if grids.index.duplicated().any():
        grids = grids.reset_index(drop=True)
    
    # Remove existing row/col from observations if present
    if 'row' in obs.columns and 'col' in obs.columns:
        logger.info("Removing existing row/col columns from observations before spatial join")
        obs = obs.drop(columns=['row', 'col'])
    
    # Get model dimensions
    nrow, ncol = top.shape
    logger.info(f"Model dimensions: {nrow} rows x {ncol} columns")
    
    # Perform spatial join with better error handling
    logger.info("Performing spatial join between observations and grid")
    try:
        df_obs = gpd.sjoin_nearest(obs, grids, how='inner', max_distance=250)
        logger.info(f"Spatial join successful with {len(df_obs)} matches")
    except Exception as e:
        logger.error(f"Error in spatial join: {str(e)}")
        logger.info("Trying clip-based approach instead")
        try:
            # Try clipping observations to the grid extent
            grid_union = grids.unary_union
            obs['geometry'] = obs.geometry.buffer(50)  # Add buffer to catch nearby points
            clipped_obs = obs[obs.intersects(grid_union)].copy()
            
            if len(clipped_obs) > 0:
                logger.info(f"Clipped {len(clipped_obs)} observations within grid boundary")
                
                # Assign nearest row/col
                clipped_obs['row'] = 0
                clipped_obs['col'] = 0
                
                for idx, point in enumerate(clipped_obs.geometry):
                    # Find nearest grid cell
                    dists = grids.distance(point)
                    nearest_idx = dists.argmin()
                    clipped_obs.loc[clipped_obs.index[idx], 'row'] = grids.loc[nearest_idx, 'row']
                    clipped_obs.loc[clipped_obs.index[idx], 'col'] = grids.loc[nearest_idx, 'col']
                
                df_obs = clipped_obs
                logger.info(f"Manually assigned row/col to {len(df_obs)} observations")
            else:
                raise ValueError("No observations within grid boundary")
        except Exception as e2:
            logger.error(f"Error in clip-based approach: {str(e2)}")
            return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Check if we have valid observations
    if len(df_obs) == 0:
        logger.warning("No observations found within model domain")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    logger.info(f"Found {len(df_obs)} observations within model domain")
    
    # Ensure we have row/col columns
    if 'row' not in df_obs.columns:
        if 'row_right' in df_obs.columns:
            df_obs['row'] = df_obs['row_right']
        elif 'Row' in df_obs.columns:
            df_obs['row'] = df_obs['Row']
    
    if 'col' not in df_obs.columns:
        if 'col_right' in df_obs.columns:
            df_obs['col'] = df_obs['col_right']
        elif 'Col' in df_obs.columns:
            df_obs['col'] = df_obs['Col']
    
    # Convert row/col to integers
    for col in ['row', 'col']:
        if col in df_obs.columns:
            df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce').fillna(0).astype(int)
    
    # Final check
    if 'row' not in df_obs.columns or 'col' not in df_obs.columns:
        logger.error(f"Could not find row/col columns. Available columns: {df_obs.columns.tolist()}")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Ensure we have all required columns and they're numeric
    for col in ['SWL', 'ELEV_DEM', 'PMP_CPCITY']:
        if col not in df_obs.columns:
            if col == 'SWL':
                df_obs[col] = 20.0
            elif col == 'ELEV_DEM':
                df_obs[col] = 300.0
            elif col == 'PMP_CPCITY':
                df_obs[col] = 0.0
        
        # Convert to numeric
        df_obs[col] = pd.to_numeric(df_obs[col], errors='coerce')
    
    # Filter out NaN values
    df_obs = df_obs.dropna(subset=['SWL', 'ELEV_DEM', 'row', 'col'])
    
    # Filter out rows with non-positive SWL and ELEV_DEM
    df_obs = df_obs[(df_obs['SWL'] > 0) & (df_obs['ELEV_DEM'] > 0)]
    
    # Filter rows within model domain
    valid_mask = (
        (df_obs['row'] >= 0) & 
        (df_obs['row'] < nrow) & 
        (df_obs['col'] >= 0) & 
        (df_obs['col'] < ncol)
    )
    df_obs = df_obs[valid_mask]
    
    # Check if we have observations left
    if len(df_obs) == 0:
        logger.warning("No valid observations after filtering")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    logger.info(f"Final filtered observations: {len(df_obs)}")
    
    # Save diagnostic plot
    try:
        plt.figure(figsize=(10, 8))
        df_obs.plot(column='SWL', legend=True)
        plt.title('Well Static Water Levels')
        os.makedirs(log_dir, exist_ok=True)
        plt.savefig(f'{log_dir}/{MODEL_NAME}_obs_heads.png')
        plt.close()
    except Exception as e:
        logger.error(f"Error creating diagnostic plot: {str(e)}")
    
    # Extract row and column arrays
    rows = df_obs['row'].values.astype(int)
    cols = df_obs['col'].values.astype(int)
    
    # Calculate heads
    logger.info("Calculating head values")
    heads = fitToMeter * (df_obs['ELEV_DEM'].values - df_obs['SWL'].values)
    
    # Remove invalid heads
    valid_head_mask = ~np.isnan(heads)
    if not np.all(valid_head_mask):
        heads = heads[valid_head_mask]
        rows = rows[valid_head_mask]
        cols = cols[valid_head_mask]
        df_obs = df_obs[valid_head_mask].copy()
    
    if len(df_obs) == 0:
        logger.warning("No valid heads after filtering NaN values")
        return create_dummy_well_data(mf, top, active), None, pd.DataFrame({'wellid': ['dummy']})
    
    # Determine layer for each well
    logger.info("Assigning wells to model layers")
    
    # Create mapping from z_botm to well locations
    z_botm_well_loc = np.zeros((len(z_botm), len(rows)))
    
    # Fill the array with bottom elevations
    for i in range(len(z_botm)):
        for j, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= r < z_botm[i].shape[0] and 0 <= c < z_botm[i].shape[1]:
                z_botm_well_loc[i, j] = z_botm[i][r, c]
            else:
                z_botm_well_loc[i, j] = np.nan
    
    # For NaN values, replace with z_botm array mean
    for i in range(len(z_botm)):
        nan_mask = np.isnan(z_botm_well_loc[i])
        if np.any(nan_mask):
            z_botm_well_loc[i, nan_mask] = np.nanmean(z_botm[i])
    
    # Find first layer where head is below layer bottom
    heads_below_layer = heads[None, :] <= z_botm_well_loc
    layer = np.argmax(heads_below_layer, axis=0)
    
    # If head not below any layer, assign to top layer
    mask_no_layer = ~np.any(heads_below_layer, axis=0)
    layer[mask_no_layer] = 0
    
    # Assign layers to DataFrame
    df_obs['layer'] = layer
    
    # Create pumping rates
    pumping_rates = -1 * gpm_to_cmd * df_obs['PMP_CPCITY'].values
    
    # Fix NaN or zero pumping rates
    pumping_rates = np.where(np.isnan(pumping_rates) | (pumping_rates == 0), -0.1, pumping_rates)
    
    # Create well data
    well_data = []
    for i, (lay, r, c, q) in enumerate(zip(layer, rows, cols, pumping_rates)):
        if 0 <= lay < len(z_botm) + 1 and 0 <= r < nrow and 0 <= c < ncol:
            well_data.append((lay, r, c, q))
    
    # Create well data for stress periods
    wel_data = {per: well_data for per in range(mf.nper)}
    
    # Create observation data
    obs_data = []
    for _, row in df_obs.iterrows():
        obs = create_obs_data(row, top, mf, fitToMeter)
        if obs is not None:
            obs_data.append(obs)
    
    logger.info(f"Created {len(well_data)} wells and {len(obs_data)} observation points")
    
    return wel_data, obs_data, df_obs

def create_dummy_well_data(mf, top, active):
    """
    Create dummy well data for testing when real data is not available.
    
    Parameters:
    -----------
    mf : flopy.modflow.Modflow
        MODFLOW model object
    top : numpy.ndarray
        Top elevation array
    active : numpy.ndarray
        Active cells array
        
    Returns:
    --------
    dict
        Dictionary of stress period data for MODFLOW well package
    """
    logger.warning("Creating dummy well data for testing")
    
    # Get model dimensions
    nlay, nrow, ncol = mf.nlay, top.shape[0], top.shape[1]
    
    # Create a small number of test wells in active cells
    well_data = []
    active_cells = np.where(active[0] > 0)
    
    # Select up to 5 active cells, evenly distributed
    if len(active_cells[0]) > 0:
        indices = np.linspace(0, len(active_cells[0])-1, min(5, len(active_cells[0]))).astype(int)
        for i in indices:
            row = active_cells[0][i]
            col = active_cells[1][i]
            # Add a small test well with minimal pumping rate
            well_data.append((0, row, col, -0.1))  # Layer 0, minimal pumping rate
    else:
        # If no active cells, place one well in the center
        row, col = nrow//2, ncol//2
        well_data.append((0, row, col, -0.1))
    
    logger.info(f"Created {len(well_data)} dummy wells for testing")
    
    # Return well data for all stress periods
    return {per: well_data for per in range(mf.nper)}

