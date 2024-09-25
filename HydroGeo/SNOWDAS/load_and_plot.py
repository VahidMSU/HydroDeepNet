import arcpy
from arcpy.sa import ExtractByMask
import numpy as np
import pandas as pd
from datetime import datetime
import os
import geopandas as gpd
import itertools
import netCDF4 as nc
from shapely import Point

class SnowDataProcessor:
	def __init__(self, base_path):
		self.base_path = base_path
		self.datasets = {
			'average_temperature': os.path.join(base_path, 'snow/SNODAS_Modeled_average_temperature_constrained_2004_2023.nc'),
			'blowing_snow_sublimation_rate': os.path.join(base_path, 'snow/SNODAS_Modeled_blowing_snow_sublimation_rate_constrained_2004_2023.nc'),
			'melt_rate': os.path.join(base_path, 'snow/SNODAS_Modeled_melt_rate_constrained_2004_2023.nc'),
			'snow_layer_thickness': os.path.join(base_path, 'snow/SNODAS_Modeled_snow_layer_thickness_constrained_2004_2023.nc'),
			'snow_water_equivalent': os.path.join(base_path, 'snow/SNODAS_Modeled_snow_water_equivalent_constrained_2004_2023.nc'),
			'snowpack_sublimation_rate': os.path.join(base_path, 'snow/SNODAS_Modeled_snowpack_sublimation_rate_constrained_2004_2023.nc'),
			'non_snow_accumulation': os.path.join(base_path, 'snow/SNODAS_Non_snow_accumulation_constrained_2004_2023.nc'),
			'snow_accumulation': os.path.join(base_path, 'snow/SNODAS_Snow_accumulation_constrained_2004_2023.nc')
		}
		self.RESOLUTION = 30

	def harnessing_raster(self, raster_path):
		reference_path = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{self.RESOLUTION}m.tif"
		arcpy.env.cellSize = reference_path
		reference_raster = arcpy.Raster(reference_path)
		arcpy.env.snapRaster = reference_path
		arcpy.env.extent = arcpy.Describe(reference_path).extent
		
		raster = arcpy.Raster(raster_path)
		workspace = "/data/MyDataBase/SWATGenXAppData/snow/"
		arcpy.env.workspace = workspace
		
		# Check if padding is needed based on reference raster dimensions
		if (raster.extent.width < reference_raster.extent.width or
			raster.extent.height < reference_raster.extent.height):
			pad_x = (reference_raster.extent.width - raster.extent.width) / 2
			pad_y = (reference_raster.extent.height - raster.extent.height) / 2
			# Calculate padding amounts
			pad_x_left = int(np.floor(pad_x))
			pad_x_right = int(np.ceil(pad_x))
			pad_y_bottom = int(np.floor(pad_y))
			pad_y_top = int(np.ceil(pad_y))
			
			# Convert raster to numpy array and pad
			arr = arcpy.RasterToNumPyArray(raster, nodata_to_value=np.nan)
			padded_arr = np.pad(arr, ((pad_y_bottom, pad_y_top), (pad_x_left, pad_x_right)), 
								mode='constant', constant_values=np.nan)
			
			# Create a new raster from the padded array
			new_lower_left = arcpy.Point(raster.extent.XMin - pad_x_left * reference_raster.meanCellWidth,
										raster.extent.YMin - pad_y_bottom * reference_raster.meanCellHeight)
			padded_raster = arcpy.NumPyArrayToRaster(padded_arr, new_lower_left,
													reference_raster.meanCellWidth, reference_raster.meanCellHeight)
			padded_raster_path = os.path.join(workspace, f'padded_{os.path.basename(raster_path)}')
			padded_raster.save(padded_raster_path)
			raster = arcpy.Raster(padded_raster_path)  # Update raster variable to the new padded raster

		# Extract and resample raster
		output_path = os.path.join(workspace, f'{os.path.basename(raster_path[:-4])}_{self.RESOLUTION}m.tif')

		ExtractByMask(raster, reference_path).save(output_path)


	def load_variable(self, file_path, var_name):
		with nc.Dataset(file_path) as dataset:
			return dataset.variables[var_name][:]

	def loading_dataset(self):
		# Load latitudes and longitudes
		SNODAS_stations = gpd.read_file(os.path.join(self.base_path, 'snow/SNODAS_locations.shp'))
		latitudes = np.loadtxt(os.path.join(self.base_path, 'snow/SNODAS_latitudes_michigan.txt'))
		longitudes = np.loadtxt(os.path.join(self.base_path, 'snow/SNODAS_longitudes_michigan.txt'))

		# Load data from NetCDF files
		variables = {key: self.load_variable(path, 'value') for key, path in self.datasets.items()}

		return variables, longitudes, latitudes, SNODAS_stations, self.datasets






	def create_raster(self, variables, longitudes, latitudes, SNODAS_stations):
		variable_names = ['average_temperature',
						  'melt_rate',
						  'snow_layer_thickness',
						  'snow_water_equivalent',
						  'snowpack_sublimation_rate',
						  'non_snow_accumulation',
						  'snow_accumulation']

		converters = [1, 1 / 100, 1, 1, 1 / 100, 1 / 10, 1 / 10]

		units = ['Kelvin',
				 'mm',
				 'mm',
				 'mm',
				 'mm',
				 'mm',
				 'kg/sqm',
				 'kg/sqm']

		# Define the start year and the total number of days
		start_year = 2000
		for variable_name, converter, unit in zip(variable_names, converters, units):
			# 'melt_rate' is a 3D array with shape (days, height, width)
			melt_rate = converter * variables[variable_name][:]  # Example array

			num_days = melt_rate.shape[0]

			# Create a date range
			dates = pd.date_range(start=datetime(start_year, 1, 1), periods=num_days)

			# Calculate the number of years
			num_years = len(np.unique(dates.year))

			# Reshape melt_rate to have years, months, and days
			annual_rate = np.empty((num_years, 12, *melt_rate.shape[1:]))
			# Calculate monthly mean melt rate
			for year, month in itertools.product(range(num_years), range(1, 13)):
				if month in [6, 7, 8, 9]:
					continue
				# Select data for the current month and year
				mask = (dates.year == start_year + year) & (dates.month == month)
				monthly_data = melt_rate[mask, :, :]

				# Check if there are any non-NaN values before calculating the mean
				if variable_name in ['average_temperature', 'melt_rate', 'snow_layer_thickness', 'snow_water_equivalent']:
					monthly_mean = (
						np.nanmean(monthly_data, axis=0)
						if np.any(~np.isnan(monthly_data))
						else np.full(melt_rate.shape[1:], np.nan)
					)
				else:
					monthly_mean = (
						np.nansum(monthly_data, axis=0)
						if np.any(~np.isnan(monthly_data))
						else np.full(melt_rate.shape[1:], np.nan)
					)

				annual_rate[year, month - 1, :, :] = monthly_mean
			# drop the 6th, 7th, 8th, 9th month
			annual_rate = np.delete(annual_rate, [5, 6, 7, 8], axis=1)
			# Calculate annual mean melt rate
			with np.errstate(invalid='ignore'):
				annual_mean_melt_rate = np.nanmean(annual_rate, axis=(0, 1))

			# Create raster dataset using arcpy
			arcpy.env.overwriteOutput = True
			arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(26990)  # WGS 1984
			# Create a raster from the annual mean melt rate
			raster_path = os.path.join(self.base_path, 'snow', f'{variable_name}_raster.tif')
			SNODAS_stations = SNODAS_stations.to_crs(epsg=26990)
			# x_cell_size = 500
			#  y_cell_size = 500
			lower_left = arcpy.Point(SNODAS_stations.geometry.x.min(), SNODAS_stations.geometry.y.min())
			## estimate the cell size based on x and y
			x_cell_size = (SNODAS_stations.geometry.x.max() - SNODAS_stations.geometry.x.min()) / melt_rate.shape[2]
			y_cell_size = (SNODAS_stations.geometry.y.max() - SNODAS_stations.geometry.y.min()) / melt_rate.shape[1]
			#print(f"estimated x cell size: {x_cell_size}")
			#### reference for extent
			reference_raster = f"/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_{self.RESOLUTION}m.tif"
			arcpy.env.extent = arcpy.Describe(reference_raster).extent
			arcpy.NumPyArrayToRaster(annual_mean_melt_rate, lower_left,
									 float(x_cell_size), float(y_cell_size)).save(raster_path)
			print(f'Successfully created raster: {raster_path}')
			destination = os.path.basename(raster_path)
			raster_path = os.path.join(self.base_path, 'snow', destination)
			print(f"raster saved in {raster_path}")
			self.harnessing_raster(raster_path)

	def process_data(self):
		variables, longitudes, latitudes, SNODAS_stations, datasets = self.loading_dataset()
		self.create_raster(variables, longitudes, latitudes, SNODAS_stations)


if __name__ == "__main__":
	BASE_PATH = 'D:/MyDataBase'
	processor = SnowDataProcessor(BASE_PATH)
	processor.process_data()
