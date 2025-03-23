#import GDAL
from MODGenX.gdal_operations import gdal_sa as GDAL
from MODGenX.Logger import Logger
import geopandas as gpd
import pandas as pd
import warnings
import flopy
import pyproj
import shutil
import numpy as np
import matplotlib.pyplot as plt
from MODGenX.utils import (generate_raster_paths, load_raster, 
	match_raster_dimensions, active_domain,
remove_isolated_cells, input_Data, GW_starting_head,
model_src,  smooth_invalid_thickness, sim_obs)
from MODGenX.rivers import river_gen, river_correction
from MODGenX.lakes import lakes_to_drain
from MODGenX.visualization import plot_data, create_plots_and_return_metrics
from MODGenX.zonation import create_error_zones_and_save
from MODGenX.well_info import well_location, well_data_import
from MODGenX.rasterize_swat import rasterize_SWAT_features
import os
import rasterio
from osgeo import gdal, ogr
from MODGenX.well_info import create_shapefile_from_modflow_grid_arcpy
from osgeo import osr, gdal

class MODGenXCore:
	def __init__(self, username, NAME,VPUID, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME):
		self.NAME        = NAME
		self.BASE_PATH   = BASE_PATH
		self.LEVEL       = LEVEL
		self.RESOLUTION  = RESOLUTION
		self.MODEL_NAME  = MODEL_NAME
		self.SWAT_MODEL_NAME = SWAT_MODEL_NAME
		self.ML          = ML
		self.VPUID       = VPUID
		self.username    = username
		self.logger = Logger(verbose=True)
		self.raster_folder            = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/' f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/rasters_input")
		self.model_path               = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/' f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}')
		self.moflow_exe_path          = "/data/SWATGenXApp/codes/bin/modflow-nwt"
		self.swat_lake_shapefile_path = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/' f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp')
		self.ref_raster_path          = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/' f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif')
		self.subbasin_path            = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/' f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/subs1.shp")
		self.original_swat_dem = os.path.join(f'/data/SWATGenXApp/Users/{self.username}/', f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Rasters/DEM/dem.tif")
		self.shape_geometry = f"/data/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_subbasins.shp"
		self.swat_river_raster_path   = os.path.join(self.model_path, 'swat_river.tif')
		self.swat_lake_raster_path    = os.path.join(self.model_path,'lake_raster.tif')
		self.head_of_last_time_step   = os.path.join(f'/data/SWATGenXApp/Users/{self.username}', fr"SWATplus_by_VPUID/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/head_of_last_time_step.jpeg")
		self.output_heads             = os.path.join(f'/data/SWATGenXApp/Users/{self.username}',f'SWATplus_by_VPUID/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/',self.MODEL_NAME+'.hds')
		self.out_shp                  = os.path.join(self.model_path, "Grids_MODFLOW")
		self.raster_path              = os.path.join(self.raster_folder, f'{NAME}_DEM_{RESOLUTION}m.tif')
		self.basin_path               = os.path.join(self.raster_folder, 'basin_shape.shp')
		self.bound_path               = os.path.join(self.raster_folder, 'bound_shape.shp')
		self.temp_image               = f'/data/SWATGenXApp/GenXAppData/codes/MODFLOW/MODGenX/_temp/{self.NAME}_{self.MODEL_NAME}.jpeg'
		self.EPSG = "EPSG:26990"
		self.dpi = 300
		self.top = None
		self.bound_raster_path = None
		self.domain_raster_path = None
		self.RESOLUTION = RESOLUTION

	def create_DEM_raster(self):
		# We need to create the DEM raster based on the SWAT DEM shapefile and the base DEM 
		# Define the base DEM path
		GDAL.env.overwriteOutput = True

		temp_path = os.path.join(os.path.dirname(self.ref_raster_path), "dem.tif")
		# Copy the raster to the new location
		GDAL.Clip_management(self.original_swat_dem, "#", temp_path, self.shape_geometry, "0", "ClippingGeometry", "NO_MAINTAIN_EXTENT") ## NO_MAINTAIN_EXTENT means that the output raster will have the same extent as the clipped raster

		# Define the target spatial reference
		target_spatial_reference = GDAL.SpatialReference(26990)  # NAD83 / Illinois East

		# Project the raster to the target spatial reference
		projected_dem_path = os.path.join(os.path.dirname(self.ref_raster_path), "projected_dem.tif")
		GDAL.ProjectRaster_management(temp_path, projected_dem_path, target_spatial_reference)

		# If the resampled file already exists, delete it
		if os.path.exists(temp_path):
			GDAL.Delete_management(temp_path)

		# Resample the raster to resolution
		GDAL.Resample_management(projected_dem_path, self.ref_raster_path, f"{self.RESOLUTION} {self.RESOLUTION}", "CUBIC")

		# Optionally, delete the intermediate projected raster to clean up
		GDAL.Delete_management(projected_dem_path)


	def defining_bound_and_active(self):
		Subbasin = gpd.read_file(self.subbasin_path)
		basin = Subbasin.dissolve().reset_index(drop=True)
		buffered = Subbasin.buffer(100)
		basin['geometry'] = buffered.unary_union
		basin = basin.set_geometry('geometry').copy()
		basin['Active'] = 1

		basin[['Active','geometry']].to_file(self.basin_path)
		bound = basin.boundary.copy()
		bound = bound.explode(index_parts=False)
		bound = bound[bound.length == bound.length.max()]
		bound = bound.buffer(self.RESOLUTION)
		bound = gpd.GeoDataFrame(geometry=bound).to_crs(self.EPSG)
		bound['Bound'] = 2

		bound[['Bound','geometry']].to_file(self.bound_path)


		# Get reference raster info first
		with rasterio.open(self.ref_raster_path) as ref_src:
			ref_shape = ref_src.shape
			ref_transform = ref_src.transform
			ref_crs = ref_src.crs
			ref_res = ref_src.res
			self.logger.info(f"Reference raster shape: {ref_shape}, resolution: {ref_res}")

		GDAL.env.workspace = self.raster_folder
		GDAL.env.overwriteOutput = True
		GDAL.env.snapRaster = self.ref_raster_path
		GDAL.env.outputCoordinateSystem = GDAL.Describe(self.ref_raster_path).spatialReference
		GDAL.env.extent = self.ref_raster_path
		GDAL.env.nodata = 9999
		
		self.bound_raster_path = os.path.join(self.raster_folder, 'bound.tif')
		self.domain_raster_path = os.path.join(self.raster_folder, 'domain.tif')

		 # Instead of using GDAL directly, we'll use our GDAL-like wrapper
		# which has PolygonToRaster_conversion already implemented
		GDAL.PolygonToRaster_conversion(
			self.basin_path, 
			"Active", 
			self.domain_raster_path, 
			cellsize=self.RESOLUTION
		)
		self.logger.info('Basin raster is created')
		
		GDAL.PolygonToRaster_conversion(
			self.bound_path, 
			"Bound", 
			self.bound_raster_path, 
			cellsize=self.RESOLUTION
		)
		self.logger.info('Bound raster is created')

		# Now we need to make sure the output rasters match the reference raster exactly
		# We'll use GDAL to resample if needed
		with rasterio.open(self.domain_raster_path) as src:
			domain_shape = src.shape
			domain_res = src.res
			self.logger.info(f"Domain raster initial shape: {domain_shape}, resolution: {domain_res}")
		
		with rasterio.open(self.bound_raster_path) as src:
			bound_shape = src.shape
			bound_res = src.res
			self.logger.info(f"Bound raster initial shape: {bound_shape}, resolution: {bound_res}")

		# If dimensions don't match, resample to match reference
		if domain_shape != ref_shape or bound_shape != ref_shape:
			self.logger.info("Resampling rasters to match reference raster dimensions...")
			
			# Create temporary file paths
			domain_temp = os.path.join(self.raster_folder, 'domain_temp.tif')
			bound_temp = os.path.join(self.raster_folder, 'bound_temp.tif')
			
			# Use GDAL Warp to resample the rasters to match the reference
			domain_ds = gdal.Open(self.domain_raster_path)
			bound_ds = gdal.Open(self.bound_raster_path)
			ref_ds = gdal.Open(self.ref_raster_path)
			
			# Get reference geotransform and projection
			ref_geo = ref_ds.GetGeoTransform()
			ref_proj = ref_ds.GetProjection()
			ref_xsize = ref_ds.RasterXSize
			ref_ysize = ref_ds.RasterYSize
			
			# Warp domain raster to match reference
			warp_options = gdal.WarpOptions(
				width=ref_xsize,
				height=ref_ysize,
				outputBounds=(ref_geo[0], ref_geo[3] + ref_ysize * ref_geo[5], 
							 ref_geo[0] + ref_xsize * ref_geo[1], ref_geo[3]),
				dstSRS=ref_proj,
				resampleAlg=gdal.GRA_NearestNeighbour
			)
			gdal.Warp(domain_temp, domain_ds, options=warp_options)
			
			# Warp bound raster to match reference
			gdal.Warp(bound_temp, bound_ds, options=warp_options)
			
			# Close datasets
			domain_ds = None
			bound_ds = None
			ref_ds = None
			
			# Replace original rasters with resampled ones
			os.replace(domain_temp, self.domain_raster_path)
			os.replace(bound_temp, self.bound_raster_path)
			
			self.logger.info("Rasters resampled to match reference dimensions")

		# Verify the dimensions match
		with rasterio.open(self.domain_raster_path) as src:
			domain_shape = src.shape
			domain_res = src.res
			self.logger.info(f"Domain raster final shape: {domain_shape}, resolution: {domain_res}")

		with rasterio.open(self.bound_raster_path) as src:
			bound_shape = src.shape
			bound_res = src.res
			self.logger.info(f"Bound raster final shape: {bound_shape}, resolution: {bound_res}")

		# Validate that the shapes match
		assert domain_shape == ref_shape, f"Domain raster shape {domain_shape} does not match reference raster shape {ref_shape}"
		assert domain_res == ref_res, f"Domain raster resolution {domain_res} does not match reference raster resolution {ref_res}"
		assert bound_shape == ref_shape, f"Bound raster shape {bound_shape} does not match reference raster shape {ref_shape}"
		assert bound_res == ref_res, f"Bound raster resolution {bound_res} does not match reference raster resolution {ref_res}"

	def discritization_configuration(self):
		nrow, ncol = self.top.shape[0], self.top.shape[1]
		n_sublay_1 = 2                                             # Number of sub-layers in the first layer
		n_sublay_2 = 3                                             # Number of sub-layers in the second layer
		nlay = n_sublay_1 + n_sublay_2 + 1                         # Adding 1 for the bedrock layer
		k_bedrock = 1e-4                                           # bedrock hydrualic conductivity
		bedrock_thickness = 40                                     # bedrock thickness

		return nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness

	def create_modflow_model(self):
		self.create_DEM_raster()
		# Check if the directory exists
		if os.path.exists(self.model_path):
			shutil.rmtree(self.model_path)
			self.logger.info(f"Directory '{self.model_path}' has been removed.")

		os.makedirs(self.model_path)
		os.makedirs(self.raster_folder)
		self.logger.info(f'Model path: {self.model_path}')

		self.defining_bound_and_active()
		
		# Check if SWAT DEM exists
		assert os.path.exists(self.original_swat_dem), f"SWAT DEM file not found at {self.original_swat_dem}"

		# Get CRS from original SWAT DEM
		swat_dem_ds = gdal.Open(self.original_swat_dem)
		assert swat_dem_ds, f"Failed to open SWAT DEM file at {self.original_swat_dem}"
		
		
		swat_dem_proj = swat_dem_ds.GetProjection()
		swat_dem_srs = osr.SpatialReference()
		swat_dem_srs.ImportFromWkt(swat_dem_proj)
		self.logger.info(f"Original SWAT DEM CRS: {swat_dem_srs.ExportToProj4()}")
		swat_dem_ds = None
		
		# Store the CRS for reference throughout the process
		self.reference_crs = swat_dem_proj
	
		load_raster_args = {
			'LEVEL': self.LEVEL,
			'RESOLUTION': self.RESOLUTION,
			'NAME': self.NAME,
			'ref_raster': self.ref_raster_path,
			'bound_raster': self.bound_raster_path,
			'active': self.domain_raster_path,
			'MODEL_NAME': self.MODEL_NAME,
			'SWAT_MODEL_NAME': self.SWAT_MODEL_NAME,
			'username': self.username,
			'VPUID': self.VPUID,
			'reference_crs': self.reference_crs  # Add reference CRS to the arguments
		}

		raster_paths = generate_raster_paths(self.RESOLUTION, self.ML)

		# Add detailed DEM inspection before loading
		dem_path = raster_paths['DEM']
		import rasterio
		self.logger.info(f"Inspecting DEM file: {dem_path}")

		self.top = load_raster(raster_paths['DEM'], load_raster_args)

		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant"

		self.logger.info(f'Shape of top: {self.top.shape}')
		self.logger.info(f'Top elevation range: min={np.min(self.top)}, max={np.max(self.top)}, mean={np.mean(self.top):.2f}')
		
		# Print histogram of values to identify potential issues
		hist, bins = np.histogram(self.top, bins=10)
		self.logger.info(f'Top elevation histogram: {hist}')
		self.logger.info(f'Top elevation histogram bins: {bins}')
		
		basin = load_raster(self.domain_raster_path, load_raster_args)
		self.logger.info(f'Shape of basin: {basin.shape}')
		
		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant"
		self.top = match_raster_dimensions(basin,self.top)

		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant"	


		assert self.top.shape == basin.shape, f"Shape of top raster {self.top.shape} does not match the shape of basin raster {basin.shape}"

		self.logger.info(f'Shape of top after matching dimensions: {self.top.shape}')

		if os.path.exists(self.swat_lake_shapefile_path):
			lake_flag = True
			rasterize_SWAT_features(self.BASE_PATH, "lakes", self.swat_lake_raster_path,  load_raster_args)
		else:
			self.logger.info('NO LAKE found in the model domain')
			lake_flag = False

		self.logger.info(f'SWAT river raster path: {self.swat_river_raster_path}')
		rasterize_SWAT_features(self.BASE_PATH, "rivers", self.swat_river_raster_path, load_raster_args)

		nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness = self.discritization_configuration()

		self.logger.info(f"Lake status: {lake_flag}")
		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant"	
		active, lake_raster = active_domain(self.top, nlay, self.swat_lake_raster_path, self.swat_river_raster_path, load_raster_args, lake_flag, fitToMeter = 0.3048)

		z_botm, k_horiz, k_vert ,recharge_data, SWL, head = input_Data (
			active, self.top, load_raster_args,
			n_sublay_1,
			n_sublay_2,
			k_bedrock,
			bedrock_thickness, self.ML )

		ibound = remove_isolated_cells(active, load_raster_args)

		strt=GW_starting_head(
			active,n_sublay_1,
			n_sublay_2,z_botm,self.top,
			head, nrow, ncol
		)

		if lake_flag:
			drain_cells = lakes_to_drain(self.swat_lake_raster_path, self.top, k_horiz, load_raster_args)

		swat_river = river_correction(
			self.swat_river_raster_path, load_raster_args, basin, active
		)
		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant"
		river_data = river_gen(nrow, ncol, swat_river, self.top, ibound)
		src,delr, delc = model_src(raster_paths['DEM'])

		os.makedirs(self.model_path, exist_ok=True)
		moflow_exe_path = os.path.join("/data/SWATGenXApp/codes/bin/", "modflow-nwt")
		shutil.copy2(moflow_exe_path, self.model_path)

		assert os.path.exists(moflow_exe_path), f"MODFLOW executable not found at {moflow_exe_path}"

		assert os.path.exists(os.path.join(self.model_path, "modflow-nwt")), "MODFLOW executable not found in the model path"

		mf = flopy.modflow.Modflow(                                                       		                      ## model object
			self.MODEL_NAME,
			exe_name=self.moflow_exe_path,
			model_ws=self.model_path,
			version='mfnwt'
		)

		dis = flopy.modflow.ModflowDis(                    										                      ## dis package
			mf, nlay, nrow, ncol,                          				# number of layers, rows, columns
			delr=delr, delc=delc,						   				# cell size
			top=self.top, botm=z_botm, 					   				# top and bottom of the model
			itmuni=4,                                      				# time unit, 4 means days
			lenuni=2,                                      				# length unit, 2 means meters
			nper=1,                                        				# number of stress periods
			perlen=[365.25],                               				# length of stress period
			nstp=[1],                                      				# number of time steps
			steady=[True],                                 				# steady state
			laycbd=[0] * (nlay - 1) + [1],                 				# confining bed
			crs = pyproj.CRS.from_user_input(self.EPSG) 				# coordinate system
		)

		mf.modelgrid.set_coord_info(xoff=src.bounds.left, yoff=src.bounds.bottom, angrot=0, crs = pyproj.CRS.from_user_input(self.EPSG))

		# Create the nwt package
		nwt = flopy.modflow.ModflowNwt(																				  ## nwt package
			mf,
			headtol=0.01,                                      			# Lower tolerance for head change
			fluxtol=0.001,                                     			# Lower tolerance for flux imbalance
			maxiterout=100,                                    			# Increase the maximum number of outer iterations
			thickfact=1e-04,                                   			# Thickness factor
			linmeth=1,                                         			# Use the GMRES linear solution method
			iprnwt=1,                                          			# self.logger.info to the MODFLOW listing file
			ibotav=0,                                          			# Do not use the option to scale the bottom layer
			options='MODERATE',                                			# Use the MODERATE option to solve the nonlinear problem
			Continue=False,                                    			# Continue the simulation if the maximum number of iterations is exceeded
			backflag=1, 		                               			# Use the Backward Formulation
			maxbackiter=5		                               			# Maximum number of iterations for the Backward Formulation
		)
		
		bas = flopy.modflow.ModflowBas(mf, ibound=np.where(ibound == 2, -1, ibound), strt=strt)           										   ## basic package

		upw = flopy.modflow.ModflowUpw(                                        										   ## upw package
			mf, hk=k_horiz, vka=k_vert,                        			# horizontal and vertical hydraulic conductivity
		    laytyp=[1]*n_sublay_1 + [0]*n_sublay_2 + [0],      			# layer type, 1 means convertible, 0 means confined
			layavg=[2] + [2] * (nlay - 1)                      			# layer average, 2 means harmonic mean
		)


		rch = flopy.modflow.ModflowRch(mf, rech=recharge_data)                   									    ## recharge package

		oc = flopy.modflow.ModflowOc(									                                                ## oc package
			mf,
			stress_period_data={(0, 0): [
				"SAVE HEAD",
				"SAVE DRAWDOWN",
				"SAVE BUDGET"
			]}, compact=False
		)

		mf.write_input()                                                                                                ## write input files

		self.logger.info("MODFLOW input files written successfully")
		
		# Create MODFLOW grid shapefile
		self.logger.info("Creating MODFLOW grid shapefile")
		grids_path = f'{self.out_shp}.geojson'
		create_shapefile_from_modflow_grid_arcpy(
			self.BASE_PATH, 
			self.model_path, 
			self.MODEL_NAME, 
			self.out_shp, 
			self.ref_raster_path
		)
		assert os.path.exists(grids_path), f"MODFLOW grid shapefile not found at {grids_path}"
		self.logger.info(f"MODFLOW grid shapefile created at {grids_path}")

		# Ensure reference raster is set correctly in load_raster_args
		load_raster_args['ref_raster'] = self.ref_raster_path
		

		self.logger.info("Importing well data")
		wel_data, obs_data, df_obs = well_data_import(
			mf, 
			self.top,
			load_raster_args,
			z_botm, 
			active, 
			grids_path,
			self.MODEL_NAME
		)

		assert len(df_obs) > 0, "No observation data found"
		
		if wel_data is None:
			self.logger.warning("No well data available - proceeding without wells")
		else:
			self.logger.info(f"Successfully imported {len(wel_data[0]) if 0 in wel_data else 0} wells")
			
			# Add well package if we have well data
			wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)
			
			# Add observation package if we have observation data
			if obs_data and len(obs_data) > 0:
				hob = flopy.modflow.ModflowHob(mf, iuhobsv=41, hobdry=0., obs_data=obs_data)
				self.logger.info(f"Added {len(obs_data)} observation wells to the model")

		rasterize_SWAT_features(self.BASE_PATH,"rivers", self.swat_river_raster_path, load_raster_args)

		swat_river = river_correction(
			self.swat_river_raster_path, load_raster_args, basin, active
		)

		self.logger.info(f"Model dimensions - rows: {nrow}, columns: {ncol}")
		self.logger.info(f"SWAT river data shape: {swat_river.shape}")
		self.logger.info(f"Top data shape: {self.top.shape}")
		self.logger.info(f"ibound data shape: {ibound.shape}")
		assert np.max(self.top) != np.min(self.top), "Top elevation data is constant - check the input data"
		river_data=river_gen(nrow, ncol, swat_river, self.top, ibound)
		self.logger.info(f"Length of river data: {len(river_data[0])}")
		riv = flopy.modflow.ModflowRiv(mf, stress_period_data=river_data)

		if lake_flag:
			rasterize_SWAT_features(self.BASE_PATH,"rivers", self.swat_river_raster_path, load_raster_args)
			drain_cells = lakes_to_drain(self.swat_lake_raster_path, self.top, k_horiz, load_raster_args)
			drn = flopy.modflow.ModflowDrn(mf, stress_period_data={0: drain_cells})
			self.logger.info(f"Added {len(drain_cells)} drain cells to represent lakes")

		mf.write_input()
		self.logger.info('Rivers are updated and model inputs written')
		check_result = mf.check()
		self.logger.info("Running MODFLOW model...")
		success, buff = mf.run_model()
		from MODGenX.visualization import plot_heads
		first_layer_simulated_head = plot_heads(self.username, self.VPUID, self.LEVEL, self.NAME, self.RESOLUTION, self.MODEL_NAME)

		#if success:
		#	self.logger.info("MODFLOW model ran successfully")
		#else:
		#	self.logger.error("MODFLOW model failed to run")
		#	self.logger.error(buff)
		#	return

		self.logger.info("Head plots generated successfully")
		assert len(df_obs) > 0, "No observation data found"
		df_sim_obs = sim_obs( self.VPUID,self.username, self.BASE_PATH, self.MODEL_NAME, mf, self.LEVEL, self.top, self.NAME, self.RESOLUTION, load_raster_args, df_obs)
		nse, mse, mae, pbias, kge = create_plots_and_return_metrics(df_sim_obs, self.username, self.VPUID, self.LEVEL, self.NAME, self.MODEL_NAME)
		self.logger.info(f"Model evaluation metrics - NSE: {nse}, MSE: {mse}, MAE: {mae}, PBIAS: {pbias}, KGE: {kge}")

		metrics = [self.MODEL_NAME, self.NAME, self.RESOLUTION, nse, mse, mae, pbias, kge]
		metrics_path = os.path.join(f'/data/SWATGenXApp/Users/{self.username}', f'SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/metrics.csv')
		with open(metrics_path, 'w') as f:
			f.write('MODEL_NAME,NAME,RESOLUTION,NSE,MSE,MAE,PBIAS,KGE\n')
			f.write(','.join(str(metric) for metric in metrics))
		self.logger.info(f"Model metrics saved to {metrics_path}")

		datasets = [
			well_location(df_sim_obs, active, str(self.NAME), self.LEVEL, self.RESOLUTION, load_raster_args),
   			smooth_invalid_thickness(self.top-strt[0]),
			strt[0] , ibound[0],    k_horiz[0] ,     k_horiz[1],       k_vert[0],          k_vert[1],
			recharge_data,swat_river,  self.top - z_botm[0], z_botm[0]-z_botm[1]
		]

		titles = ['water wells location', "SWL initial",'Head',  'Active Cells','K Horizontal 1',
				'K Horizontal 2', 'K Vertical 1', 'K Vertical 2', 'Recharge','base flow','Thickness 1', 'thickness 2']

		model_input_figure_path = f"/data/SWATGenXApp/Users/{self.username}/SWATplus_by_VPUID/{self.VPUID}/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/input_figures.jpeg"

		plot_data(datasets, titles, model_input_figure_path)
		self.logger.info(f"Model input visualizations saved to {model_input_figure_path}")

		create_error_zones_and_save(self.model_path, load_raster_args, self.ML)
		self.logger.info("Error zones created and saved successfully")
