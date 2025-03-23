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
from MODGenX.utils import (load_raster, 
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
from MODGenX.config import MODFLOWGenXPaths
from MODGenX.path_handler import PathHandler

class MODGenXCore:
	def __init__(self, username, NAME, VPUID, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME, config=None):
		self.NAME        = NAME
		self.BASE_PATH   = BASE_PATH
		self.LEVEL       = LEVEL
		self.RESOLUTION  = RESOLUTION
		self.MODEL_NAME  = MODEL_NAME
		self.SWAT_MODEL_NAME = SWAT_MODEL_NAME
		self.ML          = ML
		self.VPUID       = VPUID
		self.username    = username
		
		# Use provided config or create a new one
		if config is None:
			self.config = MODFLOWGenXPaths(
				username=username,
				BASE_PATH=BASE_PATH,
				MODFLOW_MODEL_NAME=MODEL_NAME,
				SWAT_MODEL_NAME=SWAT_MODEL_NAME,
				LEVEL=LEVEL,
				VPUID=VPUID,
				NAME=NAME,
				RESOLUTION=RESOLUTION
			)
		else:
			self.config = config
		
		# Initialize path handler
		self.path_handler = PathHandler(self.config)
		
		 # Initialize logger with path_handler
		self.logger = Logger(verbose=True, path_handler=self.path_handler)
		
		# Set commonly used paths
		self.raster_folder = self.path_handler.get_raster_input_dir()
		self.model_path = self.path_handler.get_model_path()
		self.moflow_exe_path = self.path_handler.get_modflow_exe_path()
		
		# Set references to shapefile paths
		shapefile_paths = self.path_handler.get_shapefile_paths()
		self.swat_lake_shapefile_path = shapefile_paths["lakes"]
		self.subbasin_path = shapefile_paths["subs"]
		self.shape_geometry = shapefile_paths["subbasins"]
		
		# Set raster paths
		self.ref_raster_path = self.path_handler.get_ref_raster_path()
		self.original_swat_dem = self.path_handler.get_swat_dem_path()
		self.swat_river_raster_path = self.path_handler.get_output_file('swat_river.tif')
		self.swat_lake_raster_path = self.path_handler.get_output_file('lake_raster.tif')
		
		# Set output paths
		self.head_of_last_time_step = self.path_handler.get_output_file('head_of_last_time_step.jpeg')
		self.output_heads = self.path_handler.get_output_file(f'{MODEL_NAME}.hds')
		self.out_shp = os.path.join(self.model_path, "Grids_MODFLOW")
		
		# Set working paths
		self.raster_path = self.path_handler.get_raster_input_file(f'{NAME}_DEM_{RESOLUTION}m.tif')
		self.basin_path = self.path_handler.get_raster_input_file('basin_shape.shp')
		self.bound_path = self.path_handler.get_raster_input_file('bound_shape.shp')
		self.temp_image = self.path_handler.get_temporary_path(f'{self.NAME}_{self.MODEL_NAME}.jpeg')
		
		# Set other parameters
		self.EPSG = "EPSG:26990"
		self.dpi = 300
		self.top = None
		self.bound_raster_path = self.path_handler.get_bound_raster_path()
		self.domain_raster_path = self.path_handler.get_domain_raster_path()
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
		"""
		Define the discretization configuration for the MODFLOW model using the config object.
		
		This method delegates to the utility function to avoid code duplication.
		
		Returns:
		--------
		tuple
			nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness
		"""
		from MODGenX.utils import discritization_configuration as disc_config
		return disc_config(self.top, self.config)

	def validate_data(self):
		"""
		Validate input data before model creation.
		
		Returns:
		--------
		bool
			True if validation passes, False otherwise
		"""
		validation_errors = []
		
		# Validate paths
		required_paths = [
			(self.model_path, "Model directory"),
			(self.ref_raster_path, "Reference raster"),
			(self.original_swat_dem, "SWAT DEM")
		]
		
		for path, desc in required_paths:
			if not os.path.exists(path):
				validation_errors.append(f"{desc} not found: {path}")
		
		# Validate DEM
		if self.top is not None:
			if np.isnan(self.top).any():
				validation_errors.append("DEM contains NaN values")
			if np.max(self.top) == np.min(self.top):
				validation_errors.append("DEM is constant (all values equal)")
		
		# Log validation results
		if validation_errors:
			for error in validation_errors:
				self.logger.error(f"Validation error: {error}")
			return False
		
		self.logger.info("Data validation passed successfully")
		return True

	def handle_error(self, error, context, critical=False):
		"""
		Handle errors in a consistent way.
		
		Parameters:
		-----------
		error : Exception
			The error that occurred
		context : str
			Description of what was happening when the error occurred
		critical : bool, optional
			Whether this is a critical error that should stop execution
			
		Returns:
		--------
		bool
			False if critical error, True otherwise
		"""
		error_msg = f"Error in {context}: {str(error)}"
		self.logger.error(error_msg)
		
		# Store error in a dedicated log file for the model
		error_log_path = os.path.join(self.model_path, "error_log.txt")
		try:
			os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
			with open(error_log_path, 'a') as f:
				f.write(f"{error_msg}\n")
		except Exception as e:
			self.logger.error(f"Could not write to error log: {str(e)}")
		
		if critical:
			raise RuntimeError(f"Critical error in {context}: {str(error)}")
		
		return not critical

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
	
		# Create load_raster_args using path_handler
		load_raster_args = self.path_handler.create_load_raster_args()

		# Get raster paths from path_handler
		raster_paths = self.path_handler.get_raster_paths(self.ML)

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
			headtol=self.config.headtol,                                      			# Lower tolerance for head change
			fluxtol=self.config.fluxtol,                                     			# Lower tolerance for flux imbalance
			maxiterout=self.config.maxiterout,                                    			# Increase the maximum number of outer iterations
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

		if success:
			self.logger.info("MODFLOW model ran successfully")
		else:
			self.logger.error("MODFLOW model failed to run")
			self.logger.error(buff)
			return

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
