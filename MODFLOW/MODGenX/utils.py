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
from MODGenX.logger_singleton import get_logger
import time
from MODGenX.clip_rasters import clip_raster_by_another



logger = get_logger()

debug = False

def get_row_col_from_coords(df, ref_raster_path):
    """Convert coordinates to row/col indices using a reference raster"""
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
    """Correct GW heads according to the lake locations"""
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

def cleaning(df, active):
    """Clean raster data based on active cells mask"""
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

def smooth_invalid_thickness(array, size=25):
    """Smooth out values in a 2D thickness array that are outside a given range"""
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
    """Remove isolated cells in the active 3D grid and ensure valid ibound values"""
    # Ensure path_handler is provided
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    
    # Get log directory for diagnostics
    log_dir = os.path.dirname(path_handler.get_log_path("isolated_cells"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Plot active cells
    plt.close()
    plt.imshow(active[0], cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(os.path.join(log_dir, "remove_isolated_cells_active_layer_1.png"))
    plt.close()

    # Get bound raster path from path_handler
    ibound_path = path_handler.get_bound_raster_path()
    
    # Load and process ibound
    ibound = load_raster(ibound_path, load_raster_args)
    ibound = match_raster_dimensions(active[0], ibound)

    # Where active is 1, ibound is 1
    new_ibound = np.where(active == 1, 1, ibound)
    new_ibound = np.where(new_ibound == 9999, 0, new_ibound)

    # Plot results
    plt.close()
    plt.imshow(new_ibound[0], cmap='viridis')
    plt.colorbar()
    plt.title("IBound Layer 1")
    plt.savefig(os.path.join(log_dir, "remove_isolated_cells_ibound_layer_1.png"))
    plt.close()
    
    return new_ibound

def GW_layers(thickness_1, thickness_2, n_sublay_1, n_sublay_2, bedrock_thickness, top):
    """Calculate bottom elevations for groundwater layers"""
    logger.info(f"Generating layer elevations with {n_sublay_1} sublayers in layer 1, {n_sublay_2} sublayers in layer 2")
    
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

    logger.info(f"Generated {len(z_botm)} layer bottoms from {np.mean(top):.2f}m to {np.mean(bedrock_bottom):.2f}m")
    return z_botm

def discritization_configuration(top, config=None):
    """Define discretization configuration for the MODFLOW model"""
    nrow, ncol = top.shape[0], top.shape[1]
    
    # Use configuration values if provided, otherwise use defaults
    if config:
        n_sublay_1 = config.n_sublay_1
        n_sublay_2 = config.n_sublay_2
        k_bedrock = config.k_bedrock
        bedrock_thickness = config.bedrock_thickness
    else:
        n_sublay_1 = 2  # Number of sub-layers in the first layer
        n_sublay_2 = 3  # Number of sub-layers in the second layer
        k_bedrock = 1e-4  # bedrock hydrualic conductivity
        bedrock_thickness = 40  # bedrock thickness

    nlay = n_sublay_1 + n_sublay_2 + 1  # Adding 1 for the bedrock layer

    return nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness

def match_raster_dimensions(base_raster, target_raster):
    """Match the dimensions of two rasters by padding or cropping the target raster"""
    base_shape = base_raster.shape
    target_shape = target_raster.shape

    if base_shape != target_shape:
        return padding_raster(
            base_shape, target_shape, target_raster
        )
    logger.info("Both rasters have the same dimensions.")
    return target_raster

def padding_raster(base_shape, target_shape, target_raster):
    """Pad or crop a raster to match a base shape"""
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

def active_domain(top, nlay, swat_lake_raster_path, swat_river_raster_path, load_raster_args, lake_flag):
    """Create active domain array for MODFLOW model"""
    logger.info('***********Active Domain***********')
    
    # Ensure path_handler is provided
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    fit_to_meter = path_handler.config.fit_to_meter
    
    # Get domain and bound raster paths
    domain_raster_path = path_handler.get_domain_raster_path()
    bound_raster_path = path_handler.get_bound_raster_path()
    
    # Get log directory for visualization
    log_dir = os.path.dirname(path_handler.get_log_path("active_domain"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Load domain raster
    active = load_raster(domain_raster_path, load_raster_args)
    active_shape = active.shape
    active = match_raster_dimensions(top, active)

    assert active_shape == top.shape, f"Active raster shape mismatch: {active_shape} vs {top.shape}"

    # Handle lake raster if enabled
    if lake_flag:
        lake_raster = load_raster(swat_lake_raster_path, load_raster_args)
        lake_raster = np.where(lake_raster == 9999, 0, lake_raster)
        lake_raster = np.where(lake_raster > 0, 1, 0)
    else:
        logger.info('No lake is considered for active domain')
        lake_raster = np.zeros_like(active)

    assert lake_raster.shape == active.shape, f"Lake raster shape mismatch: {lake_raster.shape} vs {active.shape}"

    # Load and process bound raster
    bound = load_raster(bound_raster_path, load_raster_args)
    bound = np.where(bound == 9999, 0, bound)
    bound = match_raster_dimensions(top, bound)
    bound = np.where(bound != 2, 0, bound)
    bound = np.where(active == 1, 1, bound)

    # Create diagnostic plots
    plt.close()
    plt.imshow(bound, cmap='viridis')
    plt.colorbar()
    plt.title("Bound Layer 1")
    plt.savefig(os.path.join(log_dir, "bound_layer.png"))
    plt.close()

    plt.imshow(active, cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(os.path.join(log_dir, "active_layer.png"))
    plt.close()

    assert bound.shape == active.shape, f"Bound raster shape mismatch: {bound.shape} vs {active.shape}"

    # Update active cells based on bound
    active = np.where(bound == 2, -1, active)

    # Update active cells with lake info if enabled
    if lake_flag:
        plt.close()
        plt.imshow(lake_raster, cmap='viridis')
        plt.colorbar()
        plt.title("Lake Layer 1")
        plt.savefig(os.path.join(log_dir, "lake_layer.png"))
        plt.close()
        active = np.where((lake_raster == 1) & (active == 1), -1, active)
    else:
        lake_raster = active.copy()

    if debug:
        logger.info(f"Raster mask: {np.sum(active == -1)}")
    
    # Create 3D active array
    active = np.repeat(active[np.newaxis, :, :], nlay, axis=0)
    active[nlay-1] = 0
    
    # Clean up possible NoData values
    active = np.where(active == 9999, 0, active)
    
    # Final diagnostic plot
    plt.imshow(active[0], cmap='viridis')
    plt.colorbar()
    plt.title("Active Layer 1")
    plt.savefig(os.path.join(log_dir, "active_layer_1_.png"))
    plt.close()
    
    return active, lake_raster

def load_raster(path, load_raster_args, BASE_PATH='/data/SWATGenXApp/GenXAppData/'):
    """Load and process a raster file"""
    assert os.path.exists(path), f"Raster file not found: {path}"
    
    # Ensure path_handler is provided
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    
    # Get parameters from path_handler
    LEVEL = path_handler.config.LEVEL
    RESOLUTION = path_handler.config.RESOLUTION
    NAME = path_handler.config.NAME
    MODEL_NAME = path_handler.config.MODFLOW_MODEL_NAME
    SWAT_MODEL_NAME = path_handler.config.SWAT_MODEL_NAME
    VPUID = path_handler.config.VPUID
    username = path_handler.config.username
    ref_raster_path = path_handler.get_ref_raster_path()
    bound_raster_path = path_handler.get_bound_raster_path()
    
    logger.info('*****************load_raster*****************')
    logger.info(f"Loading raster: {path}")
    
    # Special handling for DEM files
    if "DEM" in path:
        logger.warning(f"WARNING: Loading elevation raster {path}")
        
        # Try direct load first for DEM files
        try:
            with rasterio.open(path) as src:
                # Check for multi-band DEMs
                if src.count > 1:
                    logger.warning(f"DEM has {src.count} bands - extracting band 1 which typically contains elevation data")
                    for i in range(1, src.count + 1):
                        band_data = src.read(i)
                        band_min, band_max = band_data.min(), band_data.max()
                        logger.warning(f"Band {i} range: {band_min} to {band_max}")
                
                # Always use the first band for DEMs
                data = src.read(1)
                no_data = src.nodata if src.nodata is not None else 9999
                data_min, data_max = data.min(), data.max()
                logger.info(f"Direct load - data range: {data_min} to {data_max}")
                
                # Check if the data range looks reasonable for elevation
                if data_min >= 0 and data_max < 10000:
                    logger.info(f"Using direct DEM data with valid range")
                    data = np.where(data == no_data, 9999, data)
                    if np.all(data == 9999):
                        logger.warning(f"DEM data has all NoData values, will try alternative method")
                    else:
                        return data
                else:
                    logger.warning(f"Direct load produced unusual elevation range: {data_min} to {data_max}, trying clip method")
        except Exception as e:
            logger.warning(f"Error in direct DEM load: {str(e)}, trying clip method")
    
    # Continue with normal processing (clipping) if direct load doesn't work
    file_name = os.path.basename(path)
    file_name_without_ext = os.path.splitext(file_name)[0]

    # Prepare output path
    user_vpuid_dir = os.path.join('/data/SWATGenXApp/Users', username, 'SWATplus_by_VPUID', VPUID)
    model_input_dir = os.path.join(user_vpuid_dir, LEVEL, str(NAME), f"{MODEL_NAME}/rasters_input")
    output_clip = os.path.join(model_input_dir, f"{NAME}_{file_name_without_ext}.tif")
    os.makedirs(model_input_dir, exist_ok=True)

    # Skip clipping for reference and bound rasters
    if path != ref_raster_path and path != bound_raster_path:
        clip_raster_by_another(BASE_PATH, path, ref_raster_path, output_clip)
        
        # Handle multi-band rasters - extract first band only
        with rasterio.open(output_clip) as src:
            count = src.count
            if count > 1:
                logger.warning(f"WARNING: Clipped raster has {count} bands - extracting first band")
                # Create a temporary file path
                temp_output = output_clip + "_temp.tif"
                
                # Use gdal to extract just the first band
                from osgeo import gdal
                try:
                    gdal.Translate(temp_output, output_clip, bandList=[1])
                    # Replace the original file with the single-band version
                    os.remove(output_clip)
                    os.rename(temp_output, output_clip)
                    logger.info(f"Successfully created single-band raster from band 1")
                except Exception as e:
                    logger.error(f"Error extracting band from multi-band raster: {str(e)}")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
            
            # Re-open the raster to get the data
            src = rasterio.open(output_clip)
            data = src.read(1)
            no_data = src.nodata if src.nodata is not None else 9999
            data_min, data_max = data.min(), data.max()
            logger.info(f"Clipped raster data range: {data_min} to {data_max}")
            data = np.where(data == no_data, 9999, data)
            
            # Special handling for DEM files with extreme values
            if "DEM" in path and (data_min < -900000 or data_max > 900000):
                logger.error(f"ERROR: Extreme values in clipped DEM. Attempting recovery.")
                try:
                    with rasterio.open(path) as orig_src:
                        orig_data = orig_src.read(1)
                        orig_min, orig_max = orig_data.min(), orig_data.max()
                        logger.info(f"Original DEM range: {orig_min} to {orig_max}")
                        if orig_min > -900000 and orig_max < 900000:
                            logger.info(f"Using original DEM data with valid range")
                            data = np.where(orig_data == orig_src.nodata, 9999, orig_data)
                except Exception as e:
                    logger.error(f"Error trying to recover original DEM data: {str(e)}")
            
            # Final validation
            if np.all(data == 9999) or np.all(data < -900000) or np.all(data > 900000):
                logger.error(f"ERROR: All values in raster are invalid!")
                raise ValueError(f"Invalid raster data in {output_clip}")
                
            return data
    else:
        # Direct loading for reference and bound rasters
        with rasterio.open(path) as src:
            count = src.count
            if count > 1:
                logger.warning(f"WARNING: Source raster {path} has {count} bands - this is unexpected")
                for i in range(1, count+1):
                    band_min = src.read(i).min()
                    band_max = src.read(i).max()
                    logger.warning(f"Band {i} range: {band_min} to {band_max}")
            
            # Always use the first band
            data = src.read(1)
            no_data = src.nodata if src.nodata is not None else 9999
            data_min, data_max = data.min(), data.max()
            logger.info(f"Reference/bound raster data range: {data_min} to {data_max}")
            data = np.where(data == no_data, 9999, data)
            return data

def model_src(DEM_path):
    """Get source characteristics from a DEM file"""
    src = rasterio.open(DEM_path)
    delr, delc = src.transform[0], -src.transform[4]
    return(src, delr, delc)

def input_Data(active, top, load_raster_args, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness, ML):
    """Prepare input data for the MODFLOW model"""
    # Ensure path_handler is provided
    assert 'path_handler' in load_raster_args, "path_handler is required in load_raster_args"
    path_handler = load_raster_args['path_handler']
    fitToMeter = path_handler.config.fit_to_meter
    recharge_conv_factor = path_handler.config.recharge_conv_factor
    logger.info(f"Using fit_to_meter value from config: {fitToMeter}")
    logger.info(f"Using recharge conversion factor from config: {recharge_conv_factor}")

    # Get raster paths from path_handler
    raster_paths = path_handler.get_raster_paths(ML)
    
    # Log all available raster paths for debugging
    logger.info(f"Available raster path keys: {list(raster_paths.keys())}")
    
    # Check if required rasters exist
    required_rasters = ["SWL", "recharge_data", "k_horiz_1", "k_horiz_2", "k_vert_1", "k_vert_2", "thickness_1", "thickness_2"]
    missing_rasters = [r for r in required_rasters if r not in raster_paths]
    
    if missing_rasters:
        logger.error(f"Missing required raster paths: {missing_rasters}")
        logger.error(f"ML setting is: {ML}")
        raise ValueError(f"Missing required raster paths: {missing_rasters}")
    
    SWL = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["SWL"], load_raster_args),active))  #### SWL
    
    # Use the recharge conversion factor from config
    recharge_data = cleaning(recharge_conv_factor*load_raster(raster_paths["recharge_data"], load_raster_args),active)  ### converting the unit from inch/year to m/day

    k_horiz_1 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_1"],load_raster_args), active))+2
    k_horiz_2 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_2"],load_raster_args),active))+2
    k_vert_1 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_1"],load_raster_args),active))+2
    k_vert_2 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_2"],load_raster_args),active))+2
    thickness_1 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_1"],load_raster_args),active))+3
    thickness_2 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_2"],load_raster_args),active))+3

    head = top-SWL

    k_horiz = [k_horiz_1] * n_sublay_1 + [k_horiz_2]*n_sublay_2 + [k_bedrock]
    k_vert = [k_vert_1]  * n_sublay_1 + [k_vert_2]*n_sublay_2 + [k_bedrock]
    z_botm = GW_layers(thickness_1, thickness_2, n_sublay_1, n_sublay_2, bedrock_thickness, top)
    
    return (z_botm, k_horiz, k_vert, recharge_data, SWL, head)
