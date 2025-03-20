import numpy as np
import flopy
import pandas as pd
import geopandas as gpd
try:
	from MODGenX.utils import *
except ImportError:
	from utils import *
def well_location(df_sim_obs, active, NAME, LEVEL, RESOLUTION, load_raster_args):
    # Assuming active is a 3D array, and we are using its first 2D slice for comparison
    # Initialize obs_array with the same shape as the first slice of active
    obs_array = np.zeros_like(active[0], dtype=float)

    obs_np = df_sim_obs[['row', 'col', 'obs_SWL_m']].to_numpy()
    print('number of observations in the original dataset:', len(obs_np))

    # Fill obs_array with SWL values using numpy array
    for row in obs_np:
        row_idx, col_idx = int(row[0]), int(row[1])
        if 0 <= row_idx < obs_array.shape[0] and 0 <= col_idx < obs_array.shape[1]:
            obs_array[row_idx, col_idx] = row[2]
        else:
            print(f"Skipping out-of-bounds index: row={row_idx}, col={col_idx}")

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

	database_paths = database_file_paths()

	obs = pd.read_pickle("{SWATGenXPaths.base_path}observations/observations_original.pk1")
	obs = obs.drop(columns=['row','col'])
	grids = pd.read_pickle(fr"{SWATGenXPaths.base_path}SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/Grids_MODFLOW.pk1")
	grids ['row']= grids['Row']
	grids ['col'] = grids['Col']
	grids = grids.drop(columns=['Row','Col'])
	df_obs= obs.sjoin(grids)
	df_obs ['wellid'] = np.arange(0,len(df_obs))

	df_obs = df_obs [(df_obs.SWL.notna()) & (df_obs.ELEV_DEM.notna()) & (df_obs.SWL <= 999) & (df_obs.PMP_CPCITY.notna()) & (df_obs.PMP_CPCITY>10) & (df_obs.WEL_STATUS=='ACT')  &  (df_obs.AQ_TYPE!='ROCK')]

	rows = df_obs['row'].values.astype(int)
	cols = df_obs['col'].values.astype(int)

	heads = fitToMeter*(df_obs['ELEV_DEM'].values - df_obs['SWL'].values)
	# Get the bottom elevations at the well locations

	print(f"Max row index: {np.max(rows)}, Max col index: {np.max(cols)}")

  #  z_botm_well_loc = np.array([z[rows, cols] for z in z_botm)
	z_botm_well_loc = np.array([z_botm[i][rows, cols] for i in range(len(z_botm))])

	# Identify the layer where the SWL is above the bottom elevation of the layer
	layer = np.argmax((heads[None, :] <= z_botm_well_loc), axis=0)
	# If SWL is below all layers, assign to the last active layer
	layer[np.all(heads[None, :] > z_botm_well_loc, axis=0)] = len(z_botm) - 2

	df_obs['layer'] = layer

	df_obs['wel_data'] = list(zip(df_obs['layer'], rows, cols, -1 * gpm_to_cmd * df_obs['PMP_CPCITY']))

	active_status = np.array([active[l][r, c] for l, r, c in zip(layer, rows, cols)])

	df_obs['is_active'] = ~np.isin(active_status, [-1,2, 0])

	df_obs = df_obs[df_obs['is_active']]

	wel_data = {per: list(df_obs['wel_data']) for per in range(mf.nper)}

	df_obs[['layer','row','col','wellid']] = df_obs[['layer','row','col','wellid']].astype(int)
	print(f"$$$$$$$$$$$$$$$ size of df_obs: {len(df_obs)}")
	if len(df_obs) == 0:
		return None, None, None
	obs_data = df_obs[['layer','wellid','row','col','SWL']].apply(lambda row: create_obs_data(row,top,mf), axis=1).tolist()

	return wel_data, obs_data, df_obs