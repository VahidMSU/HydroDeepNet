#import arcpy
from MODGenX.gdal_operations import gdal_sa as arcpy
import geopandas as gpd
import pandas as pd
import warnings
import flopy
import pyproj
import shutil
import numpy as np
import matplotlib.pyplot as plt
from MODGenX.utils import generate_raster_paths, load_raster, match_raster_dimensions, active_domain, remove_isolated_cells, input_Data, GW_starting_head, model_src, create_shapefile_from_modflow_grid_arcpy, smooth_invalid_thickness, sim_obs
from MODGenX.rivers import river_gen, river_correction
from MODGenX.lakes import lakes_to_drain
from MODGenX.visualization import plot_data, create_plots_and_return_metrics
from MODGenX.zonation import create_error_zones_and_save
from MODGenX.well_info import well_location, well_data_import
from MODGenX.rasterize_swat import rasterize_SWAT_features
import os

class MODGenXCore:
	def __init__(self, NAME, BASE_PATH, LEVEL, RESOLUTION, MODEL_NAME, ML, SWAT_MODEL_NAME):
		self.NAME         = NAME
		self.BASE_PATH    = BASE_PATH
		self.LEVEL        = LEVEL
		self.RESOLUTION   = RESOLUTION
		self.MODEL_NAME   = MODEL_NAME
		self.SWAT_MODEL_NAME = SWAT_MODEL_NAME
		self.ML           = ML
		self.raster_folder             = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}/rasters_input")
		self.model_path                = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{MODEL_NAME}')
		self.moflow_exe_path           = os.path.join(self.model_path,"MODFLOW-NWT_64.exe")
		self.swat_lake_shapefile_path  = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_lakes.shp')
		self.ref_raster_path           = os.path.join(BASE_PATH, f'SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif')
		self.subbasin_path             = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/subs1.shp")
		self.SWAT_dem_path             = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/DEM_{RESOLUTION}m.tif")
		self.base_dem = os.path.join(BASE_PATH, f"SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Rasters/DEM/dem.tif")
		self.shape_geometry = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/{LEVEL}/{NAME}/{SWAT_MODEL_NAME}/Watershed/Shapes/SWAT_plus_subbasins.shp"
		### check if the resolution of self.SWAT_dem_path is actually the resolution
		resolution = arcpy.GetRasterProperties_management(self.SWAT_dem_path, "CELLSIZEX").getOutput(0)
		DEM_flag = bool(resolution != str(RESOLUTION))#, f"Resolution of the SWAT DEM raster is {resolution}m, not {RESOLUTION}m as expected."
		if not os.path.exists(self.SWAT_dem_path) or DEM_flag:
			# Define the base DEM path
			arcpy.env.overwriteOutput = True

			temp_path = os.path.join(os.path.dirname(self.SWAT_dem_path), "dem.tif")
			# Copy the raster to the new location
			arcpy.Clip_management(self.base_dem, "#", temp_path, self.shape_geometry, "0", "ClippingGeometry", "NO_MAINTAIN_EXTENT") ## NO_MAINTAIN_EXTENT means that the output raster will have the same extent as the clipped raster

			# Define the target spatial reference
			target_spatial_reference = arcpy.SpatialReference(26990)  # NAD83 / Illinois East

			# Project the raster to the target spatial reference
			projected_dem_path = os.path.join(os.path.dirname(self.SWAT_dem_path), "projected_dem.tif")
			arcpy.ProjectRaster_management(temp_path, projected_dem_path, target_spatial_reference)

			# If the resampled file already exists, delete it
			if os.path.exists(temp_path):
				arcpy.Delete_management(temp_path)

			# Resample the raster to resolution
			arcpy.Resample_management(projected_dem_path, self.SWAT_dem_path, f"{self.RESOLUTION} {self.RESOLUTION}", "CUBIC")

			# Optionally, delete the intermediate projected raster to clean up
			arcpy.Delete_management(projected_dem_path)

		self.swat_river_raster_path    = os.path.join(self.model_path, 'swat_river.tif')
		self.swat_lake_raster_path     = os.path.join(self.model_path,'lake_raster.tif')
		self.head_of_last_time_step    = os.path.join(self.BASE_PATH, fr"SWAT_input/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/head_of_last_time_step.jpeg")
		self.output_heads              = os.path.join(self.BASE_PATH,f'SWAT_input/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/',self.MODEL_NAME+'.hds')
		self.out_shp                   = os.path.join(self.model_path, "Grids_MODFLOW")
		self.raster_path               = os.path.join(self.raster_folder, f'{NAME}_DEM_{RESOLUTION}m.tif')
		self.basin_path                = os.path.join(self.raster_folder, 'basin_shape.shp')
		self.bound_path                = os.path.join(self.raster_folder, 'bound_shape.shp')
		self.temp_image                = f'/data2/MyDataBase/SWATGenXAppData/codes/MODFLOW/MODGenX/_temp/{self.NAME}_{self.MODEL_NAME}.jpeg'
		self.EPSG = "EPSG:26990"
		self.dpi  = 300
		self.top  = None
		self.bound_raster_path = None
		self.domain_raster_path = None
		self.RESOLUTION = RESOLUTION

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

		self.Polygon2Raster()

	def Polygon2Raster(self):

		arcpy.env.workspace = self.raster_folder
		arcpy.env.overwriteOutput = True
		reference_raster_path = os.path.join(self.BASE_PATH, f"all_rasters/DEM_{self.RESOLUTION}m.tif")
		arcpy.env.snapRaster = reference_raster_path
		arcpy.env.outputCoordinateSystem = arcpy.Describe(reference_raster_path).spatialReference
		arcpy.env.extent = self.SWAT_dem_path
		arcpy.env.nodata = np.nan
		self.bound_raster_path = os.path.join(self.raster_folder, 'bound.tif')
		self.domain_raster_path = os.path.join(self.raster_folder, 'domain.tif')
		arcpy.PolygonToRaster_conversion(self.basin_path, "Active", self.domain_raster_path, cellsize=self.RESOLUTION)
		print('basin raster is created')
		arcpy.PolygonToRaster_conversion(self.bound_path, "Bound", self.bound_raster_path, cellsize=self.RESOLUTION)
		print('bound raster is created')

	def plot_heads(self):
		"""
		This function reads a MODFLOW head binary file and creates a plot for the head data for the last time step.

		Parameters:
		cmap (str): The colormap to use for the plot. Default is 'viridis'.

		Returns:
		None.
		"""
		BASE_PATH = "/data2/MyDataBase/SWATGenXAppData/"
		# create the headfile object
		headobj = flopy.utils.binaryfile.HeadFile(self.output_heads)

		# get all of the time steps
		times = headobj.get_times()

		# Get the head data for the last time
		head = headobj.get_data(totim=times[-1], mflay=0)

		# plot the heads for the last time step
		plt.figure(figsize=(10,10))
		mask = head[:, :] > 0
		masked_data = np.ma.masked_where(~mask, head[:, :])
		plt.imshow(masked_data, cmap='viridis')
		plt.colorbar(label='Head (meters)')
		plt.title('Heads for last time step')
		plt.savefig(self.head_of_last_time_step, dpi=self.dpi)
		#shutil.copy2(self.head_of_last_time_step, self.temp_image)

		plt.close()

		return head[:, :]
	def discritization_configuration(self):
		nrow, ncol = self.top.shape[0], self.top.shape[1]
		n_sublay_1 = 2                                             # Number of sub-layers in the first layer
		n_sublay_2 = 3                                             # Number of sub-layers in the second layer
		nlay = n_sublay_1 + n_sublay_2 + 1                         # Adding 1 for the bedrock layer
		k_bedrock = 1e-4                                           # bedrock hydrualic conductivity
		bedrock_thickness = 40                                     # bedrock thickness

		return nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness

	def create_modflow_model(self):

		# Check if the directory exists
		if os.path.exists(self.model_path):
			shutil.rmtree(self.model_path)
			print(f"Directory '{self.model_path}' has been removed.")

		os.makedirs(self.model_path)
		os.makedirs(self.raster_folder)
		print(f'model path: {self.model_path}')

		self.defining_bound_and_active()

		load_raster_args = {
			'LEVEL': self.LEVEL,
			'RESOLUTION': self.RESOLUTION,
			'NAME': self.NAME,
			'ref_raster': self.ref_raster_path,
			'bound_raster': self.bound_raster_path,
			'active': self.domain_raster_path,
			'MODEL_NAME': self.MODEL_NAME,
			'SWAT_MODEL_NAME': self.SWAT_MODEL_NAME
		}

		raster_paths = generate_raster_paths(self.RESOLUTION, self.ML)

		self.top = load_raster(raster_paths['DEM'], load_raster_args)
		print(f' ############## shape of top {self.top.shape} ############## ')
		basin  = load_raster(self.domain_raster_path, load_raster_args)
		print(f' ############## shape of basin {basin.shape} ############## ')
		self.top  = match_raster_dimensions(basin,self.top)
		assert self.top.shape == basin.shape, f"Shape of top raster {self.top.shape} does not match the shape of basin raster {basin.shape}"

		print(f' ############## shape of top {self.top.shape} ############## ')

		if os.path.exists(self.swat_lake_shapefile_path):
			lake_flag = True
			rasterize_SWAT_features(self.BASE_PATH, "lakes", self.swat_lake_raster_path,  load_raster_args)
		else:
			print(' ################## NO LAKE ################### ')
			lake_flag = False

		print(self.swat_river_raster_path)
		rasterize_SWAT_features(self.BASE_PATH, "rivers", self.swat_river_raster_path, load_raster_args)

		nlay, nrow, ncol, n_sublay_1, n_sublay_2, k_bedrock, bedrock_thickness = self.discritization_configuration()

		print(f"lake status {lake_flag}")

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
		river_data = river_gen(nrow, ncol, swat_river, self.top, ibound)
		src,delr, delc = model_src(raster_paths['DEM'])

		os.makedirs(self.model_path, exist_ok=True)

		shutil.copy2(os.path.join(self.BASE_PATH,"bin/MODFLOW-NWT_64.exe"), self.model_path)

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
			iprnwt=1,                                          			# Print to the MODFLOW listing file
			ibotav=0,                                          			# Do not use the option to scale the bottom layer
			options='MODERATE',                                			# Use the MODERATE option to solve the nonlinear problem
			Continue=False,                                    			# Continue the simulation if the maximum number of iterations is exceeded
			backflag=1, 		                               			# Use the Backward Formulation
			maxbackiter=5		                               			# Maximum number of iterations for the Backward Formulation
		)
		bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)           										   ## basic package

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



		create_shapefile_from_modflow_grid_arcpy(self.BASE_PATH, self.model_path, self.MODEL_NAME, self.out_shp, self.raster_path)

		grids_path = f'{self.out_shp}.geojson'

		wel_data,obs_data, df_obs =  well_data_import(
			mf, self.top,
			load_raster_args,
			z_botm, active, grids_path,
			self.MODEL_NAME
		)

		if obs_data:
			wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)
			hob = flopy.modflow.ModflowHob(mf, iuhobsv=41, hobdry=-9999., obs_data=obs_data)

		rasterize_SWAT_features(self.BASE_PATH,"rivers", self.swat_river_raster_path, load_raster_args)

		swat_river = river_correction(
			self.swat_river_raster_path, load_raster_args, basin, active
		)

		print('nrows', nrow,'ncol', ncol)
		print(f"SWAT river data shape: {swat_river.shape}")
		print(f"Top data shape: {self.top.shape}")
		print(f"ibound data shape: {ibound.shape}")

#		print(f)

		river_data=river_gen(nrow, ncol, swat_river, self.top, ibound)
		print(f"length of river data {len(river_data[0])}")
		riv = flopy.modflow.ModflowRiv(mf, stress_period_data=river_data)

		if lake_flag:
			rasterize_SWAT_features(self.BASE_PATH,"rivers", self.swat_river_raster_path, load_raster_args)
			drain_cells = lakes_to_drain(self.swat_lake_raster_path, self.top, k_horiz, load_raster_args)
			drn = flopy.modflow.ModflowDrn(mf, stress_period_data={0: drain_cells})

		mf.write_input()

		print('rivers are updated')

		mf.check()

		success, buff = mf.run_model()


		first_layer_simulated_head = self.plot_heads()

		df_sim_obs = sim_obs (self.BASE_PATH, self.MODEL_NAME, mf, self.LEVEL, self.top, self.NAME, self.RESOLUTION, load_raster_args, df_obs)
		nse, mse, mae, pbias, kge = create_plots_and_return_metrics (df_sim_obs, self.LEVEL, self.NAME, self.MODEL_NAME)

		metrics = [self.MODEL_NAME, self.NAME, self.RESOLUTION, nse, mse, mae, pbias, kge]
		metrics_path = os.path.join(self.BASE_PATH, f'SWAT_input/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/metrics.csv')
		with open(metrics_path, 'w') as f:
			f.write('MODEL_NAME,NAME,RESOLUTION,NSE,MSE,MAE,PBIAS,KGE\n')
			f.write(','.join(str(metric) for metric in metrics))

		datasets = [
			well_location(df_sim_obs, active, str(self.NAME), self.LEVEL, self.RESOLUTION, load_raster_args),
   			smooth_invalid_thickness(self.top-strt[0]),
			strt[0] , ibound[0],    k_horiz[0] ,     k_horiz[1],       k_vert[0],          k_vert[1],
			recharge_data,swat_river,  self.top - z_botm[0], z_botm[0]-z_botm[1]
		]

		titles = ['water wells location', "SWL initial",'Head',  'Active Cells','K Horizontal 1',
				'K Horizontal 2', 'K Vertical 1', 'K Vertical 2', 'Recharge','base flow','Thickness 1', 'thickness 2']

		model_input_figure_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/{self.LEVEL}/{self.NAME}/{self.MODEL_NAME}/input_figures.jpeg"

		plot_data(datasets, titles, model_input_figure_path)

		create_error_zones_and_save(self.model_path, load_raster_args, self.ML)
