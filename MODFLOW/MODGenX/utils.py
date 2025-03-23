from MODGenX.gdal_operations import gdal_sa as GDAL
import geopandas as gpd
import rasterio
import pyproj
import numpy as np
import os
import itertools
import flopy
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from MODGenX.Logger import Logger 
import time
from utils.clip_rasters import clip_raster_by_another


logger = Logger(verbose=True)

debug = False

def get_row_col_from_coords (df, ref_raster_path):
    logger.info(f"Converting coordinates to row/col using reference raster: {ref_raster_path}")
    start_time = time.time()
    try:
        with rasterio.open(ref_raster_path) as src:
            x = df['geometry'].apply(lambda p: p.x).values
            y = df['geometry'].apply(lambda p: p.y).values
            row, col = src.index(x, y)
            df['row'], df['col'] = row, col
            logger.info(f"Coordinate conversion completed successfully. Processed {len(df)} points in {time.time() - start_time:.2f} seconds")
        return df
    except Exception as e:
        logger.error(f"Error in get_row_col_from_coords: {str(e)}")
        raise

def GW_starting_head(active, n_sublay_1, n_sublay_2, z_botm, top, head, nrow, ncol):

    """correcting GW heads according to the lake locations"""
    logger.info(f"Generating starting heads for groundwater model with {n_sublay_1 + n_sublay_2 + 1} layers")
    
    try:
        strt = [np.where((active[0,:,:]==-1) , top-1, head) for _ in range(n_sublay_1)]
        strt.extend(
            np.where((active[0, :, :] == -1), top - 1, z_botm[n_sublay_1 + i - 1])
            for i in range(n_sublay_2)
        )
        strt.append(np.full((nrow, ncol), z_botm[-2]))
        
        logger.info(f"Starting heads generated successfully: {len(strt)} layers")
        return strt
    except Exception as e:
        logger.error(f"Error in GW_starting_head: {str(e)}")
        raise

def cleaning(df,active):
    logger.info(f"Cleaning raster data with shape {df.shape}")
    try:
        mask = active[0,:,:] == 0
        if mask.shape != df.shape:
            ## use padding to match the dimensions
            logger.info(f'Padding the raster to match dimensions. Current: {df.shape}, Target: {mask.shape}')
            df = match_raster_dimensions(active[0], df)
            logger.info(f'Padding completed. New shape: {df.shape}')
        
        zeros_before = np.sum(df == 0)
        df[mask] = 0
        zeros_after = np.sum(df == 0)
        logger.info(f'Cleaning set {zeros_after - zeros_before} additional cells to zero')
        
        return df
    except Exception as e:
        logger.error(f"Error in cleaning function: {str(e)}")
        raise

def sim_obs(VPUID, username, BASE_PATH, MODEL_NAME, mf, LEVEL, top, NAME, RESOLUTION, load_raster_args,df_obs, fitToMeter = 0.3048):
    """
    Compare observed and simulated water levels.
    """
    logger.info(f"Starting sim_obs comparison for {MODEL_NAME} with {len(df_obs)} observation points")
    start_time = time.time()
    

    SDIR = f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}'
    hds_path = f'{SDIR}/{LEVEL}/{NAME}/{MODEL_NAME}/{MODEL_NAME}.hds'

    assert os.path.exists(hds_path), f"Head file not found: {hds_path}"

    # Load simulated heads
    logger.info(f"Loading head file: {hds_path}")
    headobj = flopy.utils.binaryfile.HeadFile(hds_path)
    sim_head = headobj.get_data(totim=headobj.get_times()[-1])
    logger.info(f"Loaded simulated heads with shape: {sim_head.shape}")
    
    model_top = mf.Dis.top.array
    hcon_1 = mf.upw.hk.array[0]
    
    # Initialize column for simulated SWL
    df_obs['sim_head_m'] = np.nan

    # Extract rows and columns
    rows = df_obs['row'].values.astype(int)
    cols = df_obs['col'].values.astype(int)
    df_obs.loc[:, 'top'] = model_top[rows, cols]

    df_obs.loc[:, 'sim_head_m'] = sim_head[0, rows, cols]
    df_obs.loc[:, 'sim_SWL_m'] = model_top[rows, cols] - sim_head[0, rows, cols]

    df_obs['obs_head_m'] = fitToMeter*(df_obs['ELEV_DEM'] - df_obs['SWL'])
    df_obs['obs_SWL_m'] = fitToMeter*df_obs['SWL']

    count_before = len(df_obs)
    df_obs.dropna(subset=['obs_head_m','obs_SWL_m', 'sim_head_m', 'obs_SWL_m'], inplace=True)
    count_after = len(df_obs)
    
    logger.info(f"Sim-obs comparison completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Observation points: {count_before} before filtering, {count_after} after filtering")
    
    return df_obs


def smooth_invalid_thickness(array, size=25):
    """
    This function smooths out the values in a 2D thickness array that are outside a given range
    by replacing them with the median value of their neighbours.

    Returns:
    numpy.ndarray: The smoothed array.
    """
    logger.info(f"Smoothing invalid thickness values with filter size {size}")
    array_shape = array.shape
    
    try:
        # Count original values outside normal range
        mask = array > 500
        extreme_values_count = np.sum(mask)
        logger.info(f"Found {extreme_values_count} extreme values (>500)")
        
        median = np.median(array[~mask])
        logger.info(f"Using median value: {median:.2f} for initial replacement")
        array[mask] = median
        
        mask = array > 1
        min_value = np.nanpercentile(array[mask], 0.5)
        max_value = np.nanpercentile(array[mask], 99.5)
        logger.info(f"Valid range determined: {min_value:.2f} to {max_value:.2f}")
    
        # Identify cells where array is less than min_value or greater than max_value
        invalid_cells = np.logical_or(array < min_value, array > max_value)
        invalid_count = np.sum(invalid_cells)
        logger.info(f"Found {invalid_count} values outside valid range ({invalid_count/array.size*100:.2f}%)")
        
        # Calculate the median filter of the thickness array with the given size - this performs the moving median
        logger.info("Applying median filter...")
        smoothed = median_filter(array, size)
    
        # Replace the invalid cells in array with their smoothed values
        array[invalid_cells] = smoothed[invalid_cells]
        logger.info(f"Smoothing complete. Output array shape: {array.shape}")
        
        return array
    except Exception as e:
        logger.error(f"Error in smooth_invalid_thickness: {str(e)}")
        logger.warning(f"Returning original array with shape {array_shape}")
        return array

def remove_isolated_cells(active, load_raster_args):
    """
    Remove isolated cells in the active 3D grid and ensure valid ibound values.

    Parameters:
    - active: 3D array representing active cells

    Returns:
    - new_ibound: 3D array with isolated cells removed and guaranteed active cells
    """
    import matplotlib.pyplot as plt
    plt.close()
    plt.imshow(active[0], cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/remove_isolated_cells_active_layer_1.png")
    plt.close()

    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    NAME = load_raster_args['NAME']
    MODFLOW_model_name = load_raster_args['MODEL_NAME']
    VPUID = load_raster_args['VPUID']
    username = load_raster_args['username']
    ibound_path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODFLOW_model_name}/rasters_input/bound.tif"
    ibound = load_raster(ibound_path, load_raster_args)
    ibound = match_raster_dimensions(active[0], ibound)

    ### where active is 1, ibound is 1
    new_ibound = np.where(active == 1, 1, ibound)

    new_ibound = np.where(new_ibound == 9999, 0, new_ibound)

    plt.close()
    plt.imshow(new_ibound[0], cmap='viridis')
    plt.colorbar()
    plt.title("IBound Layer 1")
    plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/remove_isolated_cells_ibound_layer_1.png")
    plt.close()


    
    return new_ibound

def GW_layers(thickness_1, thickness_2, n_sublay_1, n_sublay_2, bedrock_thickness, top):
    """
    Calculate bottom elevations for groundwater layers.

    Parameters:
    - thickness_1, thickness_2: Thickness of the first and second main layers.
    - n_sublay_1, n_sublay_2: Number of sub-layers in the first and second main layers.
    - top: Top elevation of the first layer.
    - bedrock_bottom: Bottom elevation of the bedrock layer.

    Returns:
    - z_botm: List of bottom elevations for all layers.
    """

    bedrock_bottom = top - thickness_1 - thickness_2 - bedrock_thickness  # Define the bottom of the bedrock layer

    # Calculate sub-layer thicknesses
    thickness_1_sub = thickness_1 / n_sublay_1
    thickness_2_sub = thickness_2 / n_sublay_2

    z_botm = []  # Initialize list to hold bottom elevations

    # Compute bottom elevations for sub-layers in the first layer
    for i in range(n_sublay_1):
        if i == 0:
            z_botm.append(top - thickness_1_sub)
        else:
            z_botm.append(z_botm[-1] - thickness_1_sub)

    # Compute bottom elevations for sub-layers in the second layer
    z_botm.extend(z_botm[-1] - thickness_2_sub for _ in range(n_sublay_2))
    # Append the bottom elevation for the bedrock layer
    z_botm.append(bedrock_bottom)


    return z_botm

def discritization_configuration(top):
    nrow, ncol = top.shape[0], top.shape[1]
    n_sublay_1 = 2  # Number of sub-layers in the first layer
    n_sublay_2 = 3  # Number of sub-layers in the second layer

    nlay = n_sublay_1 + n_sublay_2 + 1  # Adding 1 for the bedrock layer

    k_bedrock = 1e-4  # bedrock hydrualic conductivity
    bedrock_thickness = 40 ### bedrock

    return nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness


def match_raster_dimensions(base_raster, target_raster)->np.ndarray:
    """

    Match the dimensions of two rasters by padding or cropping the target raster.
    the rule is to crop the target raster to match the base raster shape
    
    """
    base_shape = base_raster.shape
    target_shape = target_raster.shape

    if base_shape != target_shape:
        return padding_raster(
            base_shape, target_shape, target_raster
        )
    logger.info("Both rasters have the same dimensions.")
    return target_raster


def padding_raster(base_shape, target_shape, target_raster):
    logger.info(f"Base raster shape: {base_shape}, Target raster shape: {target_shape}")

    diff_rows = base_shape[0] - target_shape[0]
    diff_cols = base_shape[1] - target_shape[1]

    if diff_rows < 0 or diff_cols < 0:
        logger.info("Base raster has smaller dimensions. Will pad the base raster instead.")
        # If base is smaller, we'll return the target raster cropped to match the base raster shape
        # This ensures we get the most important central part of the target raster
        if diff_rows < 0 and diff_cols < 0:
            # Crop the target raster to match base dimensions
            # Take the center portion of the target raster
            start_row = (target_shape[0] - base_shape[0]) // 2
            start_col = (target_shape[1] - base_shape[1]) // 2
            return target_raster[start_row:start_row + base_shape[0], start_col:start_col + base_shape[1]]
        elif diff_rows < 0:
            # Crop rows only
            start_row = (target_shape[0] - base_shape[0]) // 2
            return target_raster[start_row:start_row + base_shape[0], :]
        else:  # diff_cols < 0
            # Crop columns only
            start_col = (target_shape[1] - base_shape[1]) // 2
            return target_raster[:, start_col:start_col + base_shape[1]]

    # Create padded target raster with the same shape as base raster
    padded_target_raster = np.pad(target_raster, ((0, diff_rows), (0, diff_cols)), 'constant', constant_values=(1))

    logger.info(f"Padded target raster shape: {padded_target_raster.shape}")
    return padded_target_raster


def defining_bound_and_active(BASE_PATH, subbasin_path, raster_folder, RESOLUTION, SWAT_dem_path):

    Subbasin = gpd.read_file(subbasin_path)


    basin = Subbasin.dissolve().reset_index(drop=True)
    buffered = Subbasin.buffer(100)

    # Dissolve the buffered polygons to get a single outer boundary
    basin['geometry'] = buffered.unary_union
    basin = basin.set_geometry('geometry').copy()
    basin ['Active'] = 1
    basin_path = os.path.join(raster_folder, 'basin_shape.shp')

    basin[['Active','geometry']].to_file(basin_path)

    bound = basin.boundary.copy()
    bound = bound.explode(index_parts=False)
    bound = bound[bound.length == bound.length.max()]
    bound = bound.buffer(RESOLUTION)
    bound = gpd.GeoDataFrame(geometry=bound)
    bound.crs = basin.crs
    bound['Bound'] = 2
    bound_path = os.path.join(raster_folder, 'bound_shape.shp')
    bound[['Bound','geometry']].to_file(bound_path)

    logger.info('Generated bound shape saved to:',os.path.basename(bound_path))
    logger.info('Generated basin shape saved to:',os.path.basename(basin_path))

    env = GDAL.env  # Use gdal_sa's env class
    env.workspace = raster_folder
    env.overwriteOutput = True  # Enable overwrite
    reference_raster_path = os.path.join(BASE_PATH, f"all_rasters/DEM_{RESOLUTION}m.tif")
    env.snapRaster = reference_raster_path

    env.outputCoordinateSystem = GDAL.Describe(reference_raster_path).spatialReference
    env.extent = SWAT_dem_path
    env.nodata = 9999

    bound_raster_path = os.path.join  (raster_folder, 'bound.tif')
    domain_raster_path = os.path.join  (raster_folder, 'domain.tif')
    GDAL.PolygonToRaster_conversion(basin_path, "Active", domain_raster_path, cellsize=RESOLUTION)
    logger.info('basin raster is created')
    GDAL.PolygonToRaster_conversion(bound_path, "Bound", bound_raster_path, cellsize=RESOLUTION)
    logger.info('bound raster is created')

    return domain_raster_path, bound_raster_path




def active_domain (top, nlay, swat_lake_raster_path, swat_river_raster_path, load_raster_args, lake_flag, fitToMeter = 0.3048):

    logger.info('***********Active Domain***********')
    active = load_raster_args['active']
    bound_raster_path = load_raster_args['bound_raster']
    active = load_raster(active, load_raster_args)

    active_shape = active.shape
    active = match_raster_dimensions(top,active)

    assert active_shape == top.shape, f"Active raster shape mismatch: {active_shape} vs {top.shape}"

    if lake_flag:
        lake_raster = load_raster(swat_lake_raster_path,load_raster_args)
        #lake_raster = match_raster_dimensions(active,lake_raster)
        lake_raster = np.where(lake_raster == 9999, 0, lake_raster)
        lake_raster = np.where(lake_raster > 0, 1, 0)
    else:
        logger.info('no lake is considered for active domain')

    assert lake_raster.shape == active.shape, f"Lake raster shape mismatch: {lake_raster.shape} vs {active.shape}"  


    bound = load_raster(bound_raster_path, load_raster_args)
    
    bound = np.where(bound == 9999, 0, bound)

    bound = match_raster_dimensions(top,bound)

    bound = np.where(bound != 2, 0, bound)

    bound = np.where(active == 1, 1, bound)

    ### plot bound here
    plt.close()
    plt.imshow(bound, cmap='viridis')
    plt.colorbar()
    plt.title("Bound Layer 1")
    plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/bound_layer.png")
    plt.close()

    plt.imshow(active, cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/active_layer.png")
    plt.close()


    assert bound.shape == active.shape, f"Bound raster shape mismatch: {bound.shape} vs {active.shape}"

    active = np.where(bound == 2 , -1 , active)

    if lake_flag:
        ## plot lake raster
        plt.close()
        plt.imshow(lake_raster, cmap='viridis')
        plt.colorbar()
        plt.title("Lake Layer 1")
        plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/lake_layer.png")
        plt.close()
        active=np.where((lake_raster==1) & (active==1), -1, active )
    else:
        lake_raster = active.copy()

    if debug: logger.info(f"raster mask:{np.sum(active==-1)}")
    active = np.repeat(active[np.newaxis, :, :], nlay, axis=0 )
    active[nlay-1] =0
# Loop through each layer and plot active domain:
    active = np.where(active == 9999, 0, active)
    plt.imshow(active[0], cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(f"/data/SWATGenXApp/codes/MODFLOW/logs/active_layer_1_.png")
    return active, lake_raster

def load_raster(path, load_raster_args, BASE_PATH='/data/SWATGenXApp/GenXAppData/'):
    assert os.path.exists(path), f"Raster file not found: {path}"
    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    NAME = load_raster_args['NAME']
    MODEL_NAME = load_raster_args['MODEL_NAME']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']
    VPUID = load_raster_args['VPUID']
    username = load_raster_args['username'] 

    ref_raster_path = load_raster_args['ref_raster']
    logger.info('*****************load_raster*****************')
    logger.info(f"Loading raster: {path}")
    if "DEM" in path:
        logger.warning(f"WARNING: Loading elevation raster {path}")
    
    # For DEM files, try to load directly without clipping first
    if "DEM" in path:
        logger.info(f"Attempting direct load of DEM file: {path}")
        with rasterio.open(path) as src:
            data = src.read(1)
            no_data = src.nodata if src.nodata is not None else 9999
            data_min, data_max = data.min(), data.max()
            logger.info(f"Direct load - data range: {data_min} to {data_max}")
            if data_min > 0 and data_max < 10000:
                logger.info(f"Using direct DEM data with valid range")
                data = np.where(data == no_data, 9999, data)
                assert not np.all(data == 9999), f"DEM data has all NoData values"
                return data
        logger.info(f"Direct load didn't produce valid elevation range, trying clip method")
    
    file_name = os.path.basename(path)
    file_name_without_ext = os.path.splitext(file_name)[0]

    output_clip = os.path.join(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}', LEVEL, str(NAME), f"{MODEL_NAME}/rasters_input", f"{NAME}_{file_name_without_ext}.tif")
    os.makedirs(os.path.join(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}', LEVEL, str(NAME), f"{MODEL_NAME}/rasters_input"), exist_ok=True)

    # Skip clipping for reference and bound rasters
    if path != load_raster_args['ref_raster'] and path != load_raster_args['bound_raster']:
        clip_raster_by_another(BASE_PATH, path, ref_raster_path, output_clip)
        with rasterio.open(output_clip) as src:
            count = src.count
            if count > 1:
                logger.warning(f"WARNING: Clipped raster has {count} bands - extracting first band")
                from osgeo import gdal
                temp_output = output_clip + "_temp.tif"
                gdal.Translate(temp_output, output_clip, bandList=[1])
                os.remove(output_clip)
                os.rename(temp_output, output_clip)
                src = rasterio.open(output_clip)
            data = src.read(1)
            no_data = src.nodata if src.nodata is not None else 9999
            data_min, data_max = data.min(), data.max()
            logger.info(f"Clipped raster data range: {data_min} to {data_max}")
            data = np.where(data == no_data, 9999, data)
            if "DEM" in path and (data_min < -900000 or data_max > 900000):
                logger.error(f"ERROR: Extreme values in clipped DEM. Attempting recovery.")
                with rasterio.open(path) as orig_src:
                    orig_data = orig_src.read(1)
                    orig_min, orig_max = orig_data.min(), orig_data.max()
                    logger.info(f"Original DEM range: {orig_min} to {orig_max}")
                    if orig_min > -900000 and orig_max < 900000:
                        logger.info(f"Using original DEM data with valid range")
                        data = np.where(orig_data == orig_src.nodata, 9999, orig_data)
            if np.all(data == 9999) or np.all(data < -900000) or np.all(data > 900000):
                logger.error(f"ERROR: All values in raster are invalid!")
                
                raise ValueError(f"Invalid raster data in {output_clip}")
            return data
    else:
        with rasterio.open(path) as src:
            count = src.count
            if count > 1:
                logger.warning(f"WARNING: Source raster {path} has {count} bands - this is unexpected")
                for i in range(1, count+1):
                    band_min = src.read(i).min()
                    band_max = src.read(i).max()
                    logger.warning(f"Band {i} range: {band_min} to {band_max}")
            data = src.read(1)
            no_data = src.nodata if src.nodata is not None else 9999
            data_min, data_max = data.min(), data.max()
            logger.info(f"Reference/bound raster data range: {data_min} to {data_max}")
            data = np.where(data == no_data, 9999, data)
            return data

def read_raster(src, arg1):
    raster = src.read(1)
    if debug:
        logger.info(f"{os.path.basename(arg1)} LOADED")
    logger.info(f"raster size loaded:{raster.shape}")
    return abs(raster)


def create_shapefile_from_modflow_grid_arcpy(BASE_PATH, model_path, MODEL_NAME, out_shp, raster_path):
    # Step 1: Read the raster to get its extent
    env = GDAL.env  # Use gdal_sa's env class
    env.workspace = BASE_PATH
    from osgeo import gdal
    RESOLUTION = 250  # Update this if your model has a different resolution
    # Check if the raster file exists
    if not os.path.exists(raster_path):
        logger.info(f"Warning: Raster file {raster_path} not found. Trying to use reference raster instead.")
        # Try to find a reference raster in the all_rasters directory
        try:
            raster_path = os.path.join(BASE_PATH, f"all_rasters/DEM_{RESOLUTION}m.tif")
            if not os.path.exists(raster_path):
                raise FileNotFoundError(f"Reference raster {raster_path} not found.")
        except NameError:
            # In case RESOLUTION is not defined in this scope
            reference_options = [250, 100, 30]
            for res in reference_options:
                raster_path = os.path.join(BASE_PATH, f"all_rasters/DEM_{res}m.tif")
                if os.path.exists(raster_path):
                    logger.info(f"Using DEM_{res}m.tif as reference raster.")
                    break
            else:
                raise FileNotFoundError("No suitable reference raster found.")
    
    # Use GDAL to get raster extent
    ds = gdal.Open(raster_path)
    if ds is None:
        raise ValueError(f"Could not open raster file: {raster_path}")
        
    gt = ds.GetGeoTransform()
    x_min_raster = gt[0]
    y_max_raster = gt[3]
    
    # Clean up
    ds = None

    # Load the model
    mf = flopy.modflow.Modflow.load(f"{MODEL_NAME}.nam", model_ws=model_path)

    # Step 2: Use the raster extent to set xoff and yoff
    xoff, yoff = x_min_raster, y_max_raster

    sr = mf.modelgrid
    angrot = sr.angrot
    epsg_code = 26990  # Update this if your model has a different EPSG code

    # Compute grid edges
    delr, delc = sr.delr, sr.delc
    xedges = np.hstack(([xoff], xoff + np.cumsum(delr)))
    yedges = np.hstack(([yoff], yoff - np.cumsum(delc)))  # Use subtraction for yedges due to y-axis convention

    # Generate arrays of vertices for all cells in the grid
    xedges, yedges = np.meshgrid(xedges, yedges)
    bottom_left = list(zip(xedges[:-1, :-1].ravel(), yedges[:-1, :-1].ravel()))
    bottom_right = list(zip(xedges[:-1, 1:].ravel(), yedges[:-1, 1:].ravel()))
    top_right = list(zip(xedges[1:, 1:].ravel(), yedges[1:, 1:].ravel()))
    top_left = list(zip(xedges[1:, :-1].ravel(), yedges[1:, :-1].ravel()))

    vertices = [list(box) for box in zip(bottom_left, bottom_right, top_right, top_left, bottom_left)]
    geoms = [Polygon(verts) for verts in vertices]

    # Add MODFLOW grid
    rows, cols = np.indices((sr.nrow, sr.ncol))
    rows_flat = rows.ravel()
    cols_flat = cols.ravel()

    # Create geodataframe
    gdf = gpd.GeoDataFrame(
        {'Row': rows_flat, 'Col': cols_flat, 'geometry': geoms},
        crs=pyproj.CRS.from_epsg(epsg_code)
    )

    # Save to shapefile
    gdf.to_file(out_shp)

    logger.info(f"Shapefile saved to {out_shp}")
    gdf.to_file(f'{out_shp}.geojson')

def model_src(DEM_path):
    src = rasterio.open(DEM_path)
    delr, delc = src.transform[0], -src.transform[4]
    return(src,delr, delc)


def generate_raster_paths(RESOLUTION,ML):

    BASE_PATH = '/data/SWATGenXApp/GenXAppData/'

    SDIR = 'all_rasters'

    if ML:
        return {
            "DEM": os.path.join(BASE_PATH, SDIR, f"DEM_{RESOLUTION}m.tif"),
            "k_horiz_1": os.path.join(BASE_PATH, SDIR, f"predictions_ML_H_COND_1_{RESOLUTION}.tif"),
            "k_horiz_2": os.path.join(BASE_PATH, SDIR, f"predictions_ML_H_COND_2_{RESOLUTION}.tif"),
            "k_vert_1": os.path.join(BASE_PATH, SDIR, f"predictions_ML_V_COND_1_{RESOLUTION}.tif"),
            "k_vert_2": os.path.join(BASE_PATH, SDIR, f"predictions_ML_V_COND_2_{RESOLUTION}.tif"),
            "thickness_1": os.path.join(BASE_PATH, SDIR, f"predictions_ML_AQ_THK_1_{RESOLUTION}.tif"),
            "thickness_2": os.path.join(BASE_PATH, SDIR, f"predictions_ML_AQ_THK_2_{RESOLUTION}.tif"),
            "recharge_data": os.path.join(BASE_PATH,SDIR ,f'Recharge_{RESOLUTION}m.tif'),
            "SWL": os.path.join(BASE_PATH, SDIR, f"kriging_output_SWL_{RESOLUTION}m.tif"),
            "k_horiz_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_H_COND_1_{RESOLUTION}m.tif"),
            "k_horiz_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_H_COND_2_{RESOLUTION}m.tif"),
            "k_vert_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_V_COND_1_{RESOLUTION}m.tif"),
            "k_vert_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_V_COND_2_{RESOLUTION}m.tif"),
            "thickness_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_AQ_THK_1_{RESOLUTION}m.tif"),
            "thickness_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_AQ_THK_2_{RESOLUTION}m.tif"),
            "SWL_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_SWL_{RESOLUTION}m.tif")
        }

    else:
        return {
            "DEM": os.path.join(BASE_PATH, SDIR, f"DEM_{RESOLUTION}m.tif"),
            "k_horiz_1": os.path.join(BASE_PATH, SDIR, f"kriging_output_H_COND_1_{RESOLUTION}m.tif"),
            "k_horiz_2": os.path.join(BASE_PATH, SDIR, f"kriging_output_H_COND_2_{RESOLUTION}m.tif"),
            "k_vert_1": os.path.join(BASE_PATH, SDIR, f"kriging_output_V_COND_1_{RESOLUTION}m.tif"),
            "k_vert_2": os.path.join(BASE_PATH, SDIR, f"kriging_output_V_COND_2_{RESOLUTION}m.tif"),
            "thickness_1": os.path.join(BASE_PATH, SDIR, f"kriging_output_AQ_THK_1_{RESOLUTION}m.tif"),
            "thickness_2": os.path.join(BASE_PATH, SDIR, f"kriging_output_AQ_THK_2_{RESOLUTION}m.tif"),
            "recharge_data": os.path.join(BASE_PATH,SDIR, f'Recharge_{RESOLUTION}m.tif'),
            "SWL": os.path.join(BASE_PATH, SDIR, f"kriging_output_SWL_{RESOLUTION}m.tif"),
            "k_horiz_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_H_COND_1_{RESOLUTION}m.tif"),
            "k_horiz_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_H_COND_2_{RESOLUTION}m.tif"),
            "k_vert_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_V_COND_1_{RESOLUTION}m.tif"),
            "k_vert_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_V_COND_2_{RESOLUTION}m.tif"),
            "thickness_1_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_AQ_THK_1_{RESOLUTION}m.tif"),
            "thickness_2_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_AQ_THK_2_{RESOLUTION}m.tif"),
            "SWL_er": os.path.join(BASE_PATH, SDIR, f"kriging_stderr_SWL_{RESOLUTION}m.tif")
            }


def generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION,username, VPUID):
    BASE_PATH = f'/data/SWATGenXApp/Users/{username}/'
    SDIR = f'SWATplus_by_VPUID/{VPUID}'
    return {

        "lakes"  : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp'),
        "rivers" : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1.shp'),
        "grids"  : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}m/Grids_MODFLOW.geojson')

    }

def database_file_paths():
    BASE_PATH = '/data/SWATGenXApp/GenXAppData/'

    return {
        "COUNTY":       os.path.join(BASE_PATH,"Well_data_krigging/Counties_dis_gr.geojson"),
        'huc12':        os.path.join(BASE_PATH,"NHDPlusData/WBDHU12/WBDHU12_26990.geojson"),
        'huc8':         os.path.join(BASE_PATH,"NHDPlusData/WBDHU8/WBDHU8_26990.geojson"),
        'huc4':         os.path.join(BASE_PATH,"NHDPlusData/WBDHU4/WBDHU4_26990.geojson"),
        'streams':      os.path.join(BASE_PATH,"NHDPlusData/streams.pkl"),
        'observations': os.path.join(BASE_PATH,"observations/observations_original.geojson"),

    }

def input_Data(active, top, load_raster_args, n_sublay_1,n_sublay_2,k_bedrock, bedrock_thickness,ML, fitToMeter=0.3048):


    RESOLUTION = load_raster_args['RESOLUTION']

    raster_paths = generate_raster_paths(RESOLUTION, ML)
    logger.info('0-########################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$')
    SWL= smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["SWL"], load_raster_args),active))  #### SWL
    logger.info('1-########################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$')
    recharge_data = cleaning((0.0254/365.25)*load_raster(raster_paths["recharge_data"], load_raster_args),active)  ### converting the unit from inch/year to m/day

    k_horiz_1  = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_1"],load_raster_args), active))+2
    k_horiz_2  = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_2"],load_raster_args),active))+2
    k_vert_1   = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_1"],load_raster_args),active))+2
    k_vert_2   = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_2"],load_raster_args),active))+2
    thickness_1 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_1"],load_raster_args),active))+3
    thickness_2 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_2"],load_raster_args),active))+3

    head = top-SWL

    k_horiz = [k_horiz_1] * n_sublay_1 + [k_horiz_2]*n_sublay_2 + [k_bedrock]
    k_vert = [k_vert_1]  * n_sublay_1 + [k_vert_2]*n_sublay_2 + [k_bedrock]
    z_botm = GW_layers (thickness_1, thickness_2, n_sublay_1, n_sublay_2, bedrock_thickness, top)
    # plt.imshow(z_botm[0])
    # plt.close()
    return (z_botm, k_horiz, k_vert ,recharge_data, SWL, head)
