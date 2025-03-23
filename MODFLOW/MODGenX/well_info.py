import numpy as np
import flopy
import pandas as pd
import geopandas as gpd
try:
    from MODGenX.utils import *
except ImportError:
    from utils import *
      
from MODGenX.logger_singleton import get_logger
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pyproj
import os
from typing import Dict, List, Tuple, Optional, Union

logger = get_logger()

def ref_raster_to_shp_grid(
    BASE_PATH: str, 
    model_path: str, 
    MODEL_NAME: str, 
    out_shp: str, 
    ref_raster_path: str
) -> gpd.GeoDataFrame:
    """Create a GeoJSON file from MODFLOW grid using the correct raster reference."""
    logger.info(f"Creating grid shapefile from model: {model_path}/{MODEL_NAME}")
    
    # Validate inputs
    if not os.path.exists(ref_raster_path):
        raise FileNotFoundError(f"Reference raster not found: {ref_raster_path}")
    
    # Get raster extent using GDAL
    from osgeo import gdal
    ds = gdal.Open(ref_raster_path)
    if ds is None:
        raise ValueError(f"Could not open raster file: {ref_raster_path}")
        
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
        {'row': rows.ravel(), 'col': cols.ravel(), 'geometry': geoms},
        crs=pyproj.CRS.from_epsg(epsg_code)
    )
    
    # Save to files
    gdf.to_file(out_shp)
    gdf.to_file(f'{out_shp}.geojson', driver='GeoJSON')
    
    logger.info(f"Shapefile saved to {out_shp} and GeoJSON to {out_shp}.geojson")
    
    return gdf


def create_obs_data(row, top, mf, fitToMeter=0.3048):
    """Create a head observation object for MODFLOW."""
    try:
        layer = int(row['layer'])
        r = int(row['row'])
        c = int(row['col'])
        well_id = str(row['WELLID'])
        
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


def well_data_import(mf, top, load_raster_args, z_botm, active, grids_path, MODEL_NAME, gpm_to_cmd=5.678):
    """
    Import and process well data for MODFLOW model.
    
    Parameters:
    -----------
    mf : flopy.modflow.Modflow
        MODFLOW model
    top : numpy.ndarray
        Top elevation array
    load_raster_args : dict
        Dictionary with raster loading arguments, must include path_handler
    z_botm : list
        List of bottom elevations for each layer
    active : numpy.ndarray
        Array of active cells
    grids_path : str
        Path to MODFLOW grid shapefile
    MODEL_NAME : str
        Name of the MODFLOW model
    gpm_to_cmd : float, optional
        Conversion factor from gallons per minute to cubic meters per day
        
    Returns:
    --------
    tuple
        (well_data, observation_data, observations_dataframe)
    """
    logger.info(f"Importing well data for model: {MODEL_NAME}")
    
    # Get path_handler from load_raster_args
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    fitToMeter = path_handler.config.fit_to_meter
    
    log_dir = os.path.dirname(path_handler.get_log_path("well_data"))

    observations_path = path_handler.get_database_file_paths()['observations']
    active_domain_path = path_handler.get_raster_input_file('basin_shape.shp')
    
    # Load observations
    active_domain_shp = gpd.read_file(active_domain_path)
    obs = gpd.read_file(observations_path) ## point
    logger.info(f"OBSERVATION COLUMNS: {obs.columns}")
    grids = gpd.read_file(grids_path)  ## polygone
    grids = grids.to_crs(obs.crs)
    active_domain_shp = active_domain_shp.to_crs(obs.crs)
    ### covert to centroid
    grids['geometry'] = grids['geometry'].centroid
    logger.info(f"GRID COLUMNS: {grids.columns}")
    
    # Validate required columns
    required_grid_columns = ["col", "row"]
    for col in required_grid_columns:
        assert col in grids.columns, f"Column '{col}' not found in grid shapefile"
    
    #df_obs = gpd.sjoin_nearest(obs, grids, how='inner', max_distance=250)
    
    ## clip operations: polygon/\polygone->point with all attributes + row/col of the polygone
    df_obs = gpd.sjoin_nearest(obs, grids, how='inner', max_distance=250)
    
    # Validate dataframe has required columns after spatial join
    required_columns = ["col", "row", "SWL", "ELEV_DEM", "PMP_CPCITY", "WELLID"]
    for col in required_columns:
        assert col in df_obs.columns, f"Column '{col}' not found in spatial join result"
    
    ## drop duplicates by averaging the values
    assert len(df_obs) > 0, "No observations found in the model domain"
    ### drop duplicates and keep the first
    df_obs = df_obs.drop_duplicates(subset=['col', 'row'], keep='first')
    ## drop "index_right" column
    if "index_right" in df_obs.columns:
        df_obs = df_obs.drop(columns=['index_right'])
    ### now clip to the active domain and remove the points outside the active domain
    df_obs = gpd.sjoin(df_obs, active_domain_shp[['geometry']], how='inner', predicate='within')
    
    # Validate dataframe still has required columns after clipping
    for col in required_columns:
        assert col in df_obs.columns, f"Column '{col}' not found after clipping to active domain"
    
    ### assert the datatype
    datatypes = {
        "col": int,
        "row": int,
        "SWL": float,
        "ELEV_DEM": float,
        "PMP_CPCITY": float,
        "WELLID": str
    }
    
    ## correct
    for col, dtype in datatypes.items():
        df_obs[col] = df_obs[col].astype(dtype)
        
    
    logger.info(f"Spatial join successful with {len(df_obs)} matches")

    
    assert len(df_obs) > 0, "No observations found in the model domain"

    # Get model dimensions for boundary checking
    nrow, ncol = top.shape
    
    # Filter observations to those within model domain
    valid_mask = (
        (df_obs['row'] >= 0) & 
        (df_obs['row'] < nrow) & 
        (df_obs['col'] >= 0) & 
        (df_obs['col'] < ncol)
    )
    df_obs = df_obs[valid_mask]


    plt.figure(figsize=(10, 8))
    df_obs.plot(column='SWL', legend=True)
    plt.title('Well Static Water Levels')
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f'{log_dir}/{MODEL_NAME}_obs_heads.png')
    plt.close()

    # Calculate head values and assign layers
    logger.info("Calculating head values")
    rows = df_obs['row'].values.astype(int)
    cols = df_obs['col'].values.astype(int)
    heads = fitToMeter * (df_obs['ELEV_DEM'].values - df_obs['SWL'].values)
    
    # Create layer mapping
    z_botm_well_loc = np.zeros((len(z_botm), len(rows)))
    
    # Fill array with bottom elevations for each layer at well locations
    for i in range(len(z_botm)):
        for j, (r, c) in enumerate(zip(rows, cols)):
            z_botm_well_loc[i, j] = z_botm[i][r, c] if 0 <= r < z_botm[i].shape[0] and 0 <= c < z_botm[i].shape[1] else np.nan
    
    # Fill NaN values with layer mean
    for i in range(len(z_botm)):
        nan_mask = np.isnan(z_botm_well_loc[i])
        if np.any(nan_mask):
            z_botm_well_loc[i, nan_mask] = np.nanmean(z_botm[i])
    
    # Find appropriate layer for each well based on head
    heads_below_layer = heads[None, :] <= z_botm_well_loc
    layer = np.argmax(heads_below_layer, axis=0)
    
    # Default to top layer if head not below any layer bottom
    mask_no_layer = ~np.any(heads_below_layer, axis=0)
    layer[mask_no_layer] = 0
    
    # Assign layers to DataFrame
    df_obs['layer'] = layer
    
    # Create pumping rates and well data
    pumping_rates = -1 * gpm_to_cmd * df_obs['PMP_CPCITY'].values
    pumping_rates = np.where(np.isnan(pumping_rates) | (pumping_rates == 0), -0.1, pumping_rates)
    
    # Create well data list
    well_data = []
    for i, (lay, r, c, q) in enumerate(zip(layer, rows, cols, pumping_rates)):
        if 0 <= lay < len(z_botm) + 1 and 0 <= r < nrow and 0 <= c < ncol:
            well_data.append((lay, r, c, q))
    
    # Create well data for stress periods
    wel_data = {per: well_data for per in range(mf.nper)}
    
    # Create observation data list
    obs_data = []
    for _, row in df_obs.iterrows():
        obs = create_obs_data(row, top, mf, fitToMeter)
        if obs is not None:
            obs_data.append(obs)
    
    logger.info(f"Created {len(well_data)} wells and {len(obs_data)} observation points")
    

    assert len(well_data) > 0, "No wells found in the model domain"
    assert len(obs_data) > 0, "No observation data found in the model domain"
    assert len(wel_data) > 0, "No well data found for stress periods"

    return wel_data, obs_data, df_obs
