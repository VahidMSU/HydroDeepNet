from MODGenX.gdal_operations import gdal_sa as arcpy
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

debug = False

def get_row_col_from_coords (df, ref_raster_path):
    with rasterio.open(ref_raster_path) as src:
        x = df['geometry'].apply(lambda p: p.x).values
        y = df['geometry'].apply(lambda p: p.y).values
        row, col = src.index(x, y)
        df['row'], df['col'] = row, col
    return df

def GW_starting_head(active, n_sublay_1, n_sublay_2, z_botm, top, head, nrow, ncol):

    """correcting GW heads according to the lake locations"""

    strt = [np.where((active[0,:,:]==-1) , top-1, head) for _ in range(n_sublay_1)]
    strt.extend(
        np.where((active[0, :, :] == -1), top - 1, z_botm[n_sublay_1 + i - 1])
        for i in range(n_sublay_2)
    )
    strt.append(np.full((nrow, ncol), z_botm[-2]))

    return(strt)

def cleaning(df,active):
    mask=active[0,:,:]==0
    if mask.shape != df.shape:
        ## use padding to match the dimensions
        print('padding the raster to match the dimensions')
        df = match_raster_dimensions(active[0],df)
        print('padding is done', "df shape:",df.shape)
    df[mask]=0
    return(df)

def sim_obs(BASE_PATH, MODEL_NAME, mf, LEVEL, top, NAME, RESOLUTION, load_raster_args,df_obs, fitToMeter = 0.3048):
    """
    Compare observed and simulated water levels.
    """
    SDIR = 'SWAT_input'
    # Load simulated heads
    headobj = flopy.utils.binaryfile.HeadFile(
        os.path.join(BASE_PATH,
            f'{SDIR}/{LEVEL}/{NAME}/{MODEL_NAME}/',
            f'{MODEL_NAME}.hds',
        )
    )
    sim_head = headobj.get_data(totim=headobj.get_times()[-1])
    model_top = mf.Dis.top.array
    hcon_1 = mf.upw.hk.array[0]
    # Initialize column for simulated SWL
    df_obs['sim_head_m'] = np.nan

    # Extract rows and columns
    rows = df_obs['row'].values.astype(int)
    cols = df_obs['col'].values.astype(int)
    df_obs.loc[:, 'top'] = model_top[rows, cols]

    df_obs.loc[:, 'sim_head_m'] = sim_head[0, rows, cols]
    df_obs.loc[:, 'sim_SWL_m'] =  model_top[rows, cols] - sim_head[0, rows, cols]

    df_obs['obs_head_m']  =  fitToMeter*(df_obs['ELEV_DEM'] - df_obs['SWL'])
    df_obs['obs_SWL_m']   =  fitToMeter*df_obs['SWL']


    df_obs.dropna(subset=['obs_head_m','obs_SWL_m' ,'sim_head_m', 'obs_SWL_m'], inplace=True)

    return df_obs

def smooth_invalid_thickness(array, size=25):
    """
    This function smooths out the values in a 2D thickness array that are outside a given range
    by replacing them with the median value of their neighbours.

    Returns:
    numpy.ndarray: The smoothed array.
    """
    mask = array>500
    median= np.median(array[~mask])
    array[mask] = median
    mask=array>1
    min_value=np.nanpercentile(array[mask], 0.5)
    max_value=np.nanpercentile(array[mask], 99.5)

    # Identify cells where array is less than min_value or greater than max_value
    invalid_cells = np.logical_or(array < min_value, array > max_value)
    #invalid_cells =  array > max_value
    # Calculate the median filter of the thickness array with the given size - this performs the moving median
    smoothed = median_filter(array, size)

    # Replace the invalid cells in array with their smoothed values
    array[invalid_cells] = smoothed[invalid_cells]

    return array

def remove_isolated_cells(active, load_raster_args):
    """
    Remove isolated cells in the active 3D grid and ensure valid ibound values.

    Parameters:
    - active: 3D array representing active cells

    Returns:
    - new_ibound: 3D array with isolated cells removed and guaranteed active cells
    """
    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    NAME = load_raster_args['NAME']

    nlay, nrow, ncol = active.shape
    new_ibound = active.copy()
    
    # Check for unexpected values in the ibound array
    unique_values = np.unique(new_ibound)
    print(f"Unique values in ibound before processing: {unique_values}")
    
    # Correct any invalid values (>1 or <-1) except for 0
    invalid_mask = (np.abs(new_ibound) > 1) & (new_ibound != 9999)
    invalid_count = np.sum(invalid_mask)
    if invalid_count > 0:
        print(f"Warning: Found {invalid_count} invalid values in ibound array. Fixing them.")
        new_ibound[invalid_mask] = 1  # Convert to active cells
    
    # Ensure -1 values are only at boundaries
    boundary_count = np.sum(new_ibound == -1)
    print(f"Boundary cells (-1) count: {boundary_count}")
    
    # Count active cells before processing
    active_count_before = np.sum(new_ibound == 1)
    print(f"Active cells (1) before processing: {active_count_before}")
    
    # Check for 9999 values (potentially NoData values from raster import)
    nodata_count = np.sum(new_ibound == 9999)
    if nodata_count > 0:
        print(f"Warning: Found {nodata_count} NoData values (9999) in ibound. Converting to 0.")
        new_ibound[new_ibound == 9999] = 0
    
    # Verify the values are now valid
    unique_values_after = np.unique(new_ibound)
    print(f"Unique values in ibound after correction: {unique_values_after}")
    
    # Remove isolated cells - only if we have enough active cells
    if active_count_before > 100:  # Arbitrary threshold
        for i, j in itertools.product(range(1, nrow-1), range(1, ncol-1)):
            if new_ibound[0, i, j] > 0:
                surrounding = new_ibound[0, i-1:i+2, j-1:j+2]
                if np.sum(surrounding > 0) < 2:  # Cell and at least one neighbor
                    for k in range(nlay):
                        new_ibound[k, i, j] = 0

    # Fix -1 values in layers below the first layer
    for i, j in itertools.product(range(nrow), range(ncol)):
        first_layer_value = new_ibound[0, i, j]
        if first_layer_value != -1:
            for k in range(1, nlay):
                if new_ibound[k, i, j] == -1:
                    new_ibound[k, i, j] = first_layer_value
    
    # Count active cells after processing
    active_count_after = np.sum(new_ibound == 1)
    print(f"Active cells (1) after processing: {active_count_after}")
    
    # If we have no active cells after processing, create a default pattern
    if active_count_after == 0:
        print("ERROR: No active cells in the ibound array! Creating a default pattern.")
        # Create a simple rectangular active domain in the center
        center_row, center_col = nrow // 2, ncol // 2
        size = min(nrow, ncol) // 3  # Use 1/3 of the smallest dimension
        
        for i in range(center_row - size, center_row + size):
            for j in range(center_col - size, center_col + size):
                if 0 <= i < nrow and 0 <= j < ncol:
                    # Don't override boundary cells
                    if new_ibound[0, i, j] != -1:
                        new_ibound[0, i, j] = 1
        
        # Propagate to lower layers (except the last one)
        for k in range(1, nlay-1):
            new_ibound[k] = new_ibound[0].copy()
            
        print(f"Created {np.sum(new_ibound == 1)} active cells in the center region.")
    
    final_unique_values = np.unique(new_ibound)
    print(f"Final unique values in ibound: {final_unique_values}")
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


def match_raster_dimensions(base_raster, target_raster):
    base_shape = base_raster.shape
    target_shape = target_raster.shape

    if base_shape != target_shape:
        return padding_raster(
            base_shape, target_shape, target_raster
        )
    print("Both rasters have the same dimensions.")
    return target_raster


def padding_raster(base_shape, target_shape, target_raster):
    print(f"Base raster shape: {base_shape}, Target raster shape: {target_shape}")

    diff_rows = base_shape[0] - target_shape[0]
    diff_cols = base_shape[1] - target_shape[1]

    if diff_rows < 0 or diff_cols < 0:
        print("Base raster has smaller dimensions. Will pad the base raster instead.")
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

    print(f"Padded target raster shape: {padded_target_raster.shape}")
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

    print('Generated bound shape saved to:',os.path.basename(bound_path))
    print('Generated basin shape saved to:',os.path.basename(basin_path))

    env = arcpy.env  # Use gdal_sa's env class
    env.workspace = raster_folder
    env.overwriteOutput = True  # Enable overwrite
    reference_raster_path = os.path.join(BASE_PATH, f"all_rasters/DEM_{RESOLUTION}m.tif")
    env.snapRaster = reference_raster_path

    env.outputCoordinateSystem = arcpy.Describe(reference_raster_path).spatialReference
    env.extent = SWAT_dem_path
    env.nodata = np.nan

    bound_raster_path  = os.path.join  (raster_folder, 'bound.tif')
    domain_raster_path = os.path.join  (raster_folder, 'domain.tif')
    arcpy.PolygonToRaster_conversion(basin_path, "Active", domain_raster_path, cellsize=RESOLUTION)
    print('basin raster is created')
    arcpy.PolygonToRaster_conversion(bound_path, "Bound", bound_raster_path, cellsize=RESOLUTION)
    print('bound raster is created')

    return domain_raster_path, bound_raster_path




def active_domain (top, nlay, swat_lake_raster_path, swat_river_raster_path, load_raster_args, lake_flag, fitToMeter = 0.3048):

    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    active = load_raster_args['active']
    bound_raster_path = load_raster_args['bound_raster']
    NAME = load_raster_args['NAME']



    print('%%%%%%%%% DEBUG $$$$$$$$$$$$$$$$$$$$$$$$$$$')
    active  = load_raster(active, load_raster_args)
    active_shape = active.shape
    active  = match_raster_dimensions(top,active)
    active_shape_after = active.shape
    if active_shape != active_shape_after:
        print('active raster is corrected to match the dimensions of the top raster')
        ### make sure the last rows and columns that are added are all zeros
        active[active_shape[0]:, :] = 0
        active[:, active_shape[1]:] = 0

    if lake_flag:
        lake_raster = load_raster(swat_lake_raster_path,load_raster_args)
        lake_raster = match_raster_dimensions(active,lake_raster)
        lake_raster[lake_raster < np.inf] = 1
    else:
        print('no lake is considered for active domain')


    bound  = load_raster(bound_raster_path, load_raster_args)
    bound  = match_raster_dimensions(top,bound)
    active = np.where(bound == 2 , -1 , active)

    if lake_flag:
        active=np.where((lake_raster==1) & (active==1), -1, active )
    else:
        lake_raster = active.copy()

    if debug: print(f"raster mask:{np.sum(active==-1)}")
    active = np.repeat(active[np.newaxis, :, :], nlay, axis=0 )
    active[nlay-1] =0
# Loop through each layer and plot active domain:
    #plt.imshow(active[0], cmap='viridis')
    #plt.colorbar()
    #plt.title("Active Layer 1")
    #plt.savefig(f"active_layer_1_{NAME}.png")
    return active, lake_raster



def load_raster(path, load_raster_args, BASE_PATH='/data2/MyDataBase/SWATGenXAppData/'):

    LEVEL = load_raster_args['LEVEL']
    RESOLUTION = load_raster_args['RESOLUTION']
    NAME = load_raster_args['NAME']
    MODEL_NAME = load_raster_args['MODEL_NAME']
    SWAT_MODEL_NAME = load_raster_args['SWAT_MODEL_NAME']

    shapefile_paths  = generate_shapefile_paths(LEVEL,NAME, SWAT_MODEL_NAME, RESOLUTION)
    database_file = database_file_paths()
    ref_raster_path = load_raster_args['ref_raster']
    print('*****************************')

    file_name = os.path.basename(path)
    # Strip the extension if it exists to avoid double extensions
    file_name_without_ext = os.path.splitext(file_name)[0]

    output_clip = os.path.join(BASE_PATH,'SWAT_input/', LEVEL, str(NAME), f"{MODEL_NAME}/rasters_input" ,f"{NAME}_{file_name_without_ext}.tif")
    os.makedirs(os.path.join(BASE_PATH,'SWAT_input/', LEVEL, str(NAME), f"{MODEL_NAME}/rasters_input") , exist_ok=True )

    if path!=load_raster_args['ref_raster'] or path!=load_raster_args['bound_raster']:
        clip_raster_by_another(BASE_PATH, path, ref_raster_path, output_clip)

        with rasterio.open(output_clip) as src:
            return read_raster(src, output_clip)
    else:

        with rasterio.open(path) as src:
            return read_raster(src, path)


def read_raster(src, arg1):
    raster = src.read(1)
    if debug:
        print(f"{os.path.basename(arg1)} LOADED")
    print("raster size loaded:",raster.shape)
    return abs(raster)


def clip_raster_by_another(BASE_PATH, raster_path, in_masking, output_path):
    """
    This function creates a new raster by masking an existing one.
    It extracts by extent and ensures the same number of rows and columns as in_masking.
    """
    env = arcpy.env  # Use gdal_sa's env class
    env.overwriteOutput = True
    os.makedirs(os.path.join("_temp/"), exist_ok=True)
    current_directory = BASE_PATH
    env.workspace = current_directory
    from osgeo import gdal
    # Get the extent and cell size of the mask raster using GDAL
    ds = gdal.Open(in_masking)
    gt = ds.GetGeoTransform()
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + gt[1] * ds.RasterXSize
    y_min = y_max + gt[5] * ds.RasterYSize
    cell_size = abs(gt[1])  # Assuming square cells
    
    # Create extent object similar to arcpy's extent
    class Extent:
        def __init__(self, xmin, ymin, xmax, ymax):
            self.XMin = xmin
            self.YMin = ymin
            self.XMax = xmax
            self.YMax = ymax
    
    extent = Extent(x_min, y_min, x_max, y_max)
    
    # Get number of rows and columns for in_masking raster
    mask_n_rows = ds.RasterYSize
    mask_n_cols = ds.RasterXSize
    
    # Clean up
    ds = None

    # Create a temporary resampled raster with the same cell size, rows, and columns as in_masking
    import uuid
    temp_resampled_raster = os.path.join("_temp", f"{str(uuid.uuid4())}.tif")
    arcpy.Resample_management(raster_path, temp_resampled_raster, cell_size, "NEAREST")

    # Extract the coordinates for the extent
    extent_str = f"{extent.XMin} {extent.YMin} {extent.XMax} {extent.YMax}"

    # Clip the resampled raster using extent
    arcpy.Clip_management(temp_resampled_raster, extent_str, output_path)

    # Optionally, you can delete the temporary resampled raster to save space
    arcpy.Delete_management(temp_resampled_raster)


def create_shapefile_from_modflow_grid_arcpy(BASE_PATH, model_path, MODEL_NAME, out_shp, raster_path):
    # Step 1: Read the raster to get its extent
    env = arcpy.env  # Use gdal_sa's env class
    env.workspace = BASE_PATH
    from osgeo import gdal
    RESOLUTION = 250  # Update this if your model has a different resolution
    # Check if the raster file exists
    if not os.path.exists(raster_path):
        print(f"Warning: Raster file {raster_path} not found. Trying to use reference raster instead.")
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
                    print(f"Using DEM_{res}m.tif as reference raster.")
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

    print(f"Shapefile saved to {out_shp}")
    gdf.to_file(f'{out_shp}.geojson')

def model_src(DEM_path):
    src = rasterio.open(DEM_path)
    delr, delc = src.transform[0], -src.transform[4]
    return(src,delr, delc)


def generate_raster_paths(RESOLUTION,ML):

    BASE_PATH = '/data2/MyDataBase/SWATGenXAppData/'

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


def generate_shapefile_paths(LEVEL, NAME, SWAT_MODEL_NAME, RESOLUTION):
    BASE_PATH = '/data2/MyDataBase/SWATGenXAppData/'
    SDIR = 'SWAT_input'
    return {

        "lakes"  : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp'),
        "rivers" : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/rivs1.shp'),
        "grids"  : os.path.join(BASE_PATH, SDIR, f'{LEVEL}/{NAME}/MODFLOW_{RESOLUTION}m/Grids_MODFLOW.geojson')

    }

def database_file_paths():
    BASE_PATH = '/data2/MyDataBase/SWATGenXAppData/'

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
    print('0-########################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$')
    SWL= smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["SWL"], load_raster_args),active))  #### SWL
    print('1-########################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$$$$$$$$$$$$$$$$$$')
    recharge_data = cleaning((0.0254/365.25)*load_raster(raster_paths["recharge_data"], load_raster_args),active)  ### converting the unit from inch/year to m/day

    k_horiz_1   = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_1"],load_raster_args), active))+2
    k_horiz_2   = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_horiz_2"],load_raster_args),active))+2
    k_vert_1    = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_1"],load_raster_args),active))+2
    k_vert_2    = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["k_vert_2"],load_raster_args),active))+2
    thickness_1 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_1"],load_raster_args),active))+3
    thickness_2 = smooth_invalid_thickness(cleaning(fitToMeter*load_raster(raster_paths["thickness_2"],load_raster_args),active))+3

    head = top-SWL

    k_horiz = [k_horiz_1] * n_sublay_1 + [k_horiz_2]*n_sublay_2 + [k_bedrock]
    k_vert  = [k_vert_1]  * n_sublay_1 + [k_vert_2]*n_sublay_2 + [k_bedrock]
    z_botm  = GW_layers (thickness_1, thickness_2, n_sublay_1, n_sublay_2, bedrock_thickness, top)
    # plt.imshow(z_botm[0])
    # plt.close()
    return (z_botm, k_horiz, k_vert ,recharge_data, SWL, head)
