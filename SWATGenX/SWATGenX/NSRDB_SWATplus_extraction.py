### the purpose of this code is to extract the SWAT PRISM locations for NSRDB extractionimport pandas as pd
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h5pyd
from functools import partial
from multiprocessing import Process
import os
import h5py
try:
	
	from SWATGenX.SWATGenXLogging import LoggerSetup
	from SWATGenX.utils import get_all_VPUIDs
except Exception:
	from SWATGenXLogging import LoggerSetup
	from utils import get_all_VPUIDs

def find_VPUID(station_no):
	from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
	CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]


def nsrdb_contructor_wrapper(variable, SWATGenXPaths, VPUID, LEVEL, NAME):
	
	output_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/PRISM/"
	shaefile_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/PRISM/PRISM_grid.shp"
	
	contructor = NSRDB_contructor(SWATGenXPaths, variable, output_path, shaefile_path)
	contructor.run()
import rasterio

class NSRDB_contructor:
	def __init__(self, SWATGenXPaths, variable, output_path, shaefile_path):
		self.output_path = output_path
		self.shapefile_path = shaefile_path
		self.SWATGenXPaths = SWATGenXPaths
		self.years  = range(2000, 2021)
		self.variable = variable

		self.data_all = None
		
		self.swat_dict = {
			'ghi': 'slr',
			'wind_speed': 'wnd',
			'relative_humidity': 'hmd',
		}
		self.logger = LoggerSetup(verbose=True, rewrite=True)
		self.logger = self.logger.setup_logger("NSRDB_contructor")
		self.logger.info(f"NSRDB_contructor: {self.variable} extraction for SWAT locations")	
	def extract_SWAT_PRISM_locations(self):
		prism_shape = gpd.read_file(self.shapefile_path)
		prism_shape['ROWCOL'] = prism_shape['row'].astype(str) + prism_shape['col'].astype(str)
		NSRD_PRISM = pd.read_pickle(self.SWATGenXPaths.NSRDB_PRISM_path)
		NSRD_PRISM['ROWCOL'] = NSRD_PRISM['row'].astype(str) + NSRD_PRISM['col'].astype(str)
		NSRDB_SWAT = pd.merge(NSRD_PRISM.drop(columns=['row','col','geometry']), prism_shape, on = "ROWCOL", how = "inner")
		NSRDB_SWAT['NSRDB_index'] = NSRDB_SWAT['NSRDB_index'].astype(int)
		self.logger.info(f"NSRDB_SWAT colums: {NSRDB_SWAT.columns.values}")

		self.nsrdb_indexes = np.unique(NSRDB_SWAT.sort_values(by ='NSRDB_index').NSRDB_index.values)
		self.logger.info(f"number of nsrdb indexes: {len(self.nsrdb_indexes)}")	
		self.NSRD_PRISM = NSRD_PRISM[NSRD_PRISM.NSRDB_index.isin(np.unique(NSRDB_SWAT.NSRDB_index.values))]
		self.logger.info(f"number of nsrdb indexes to be extracted: {len(self.NSRD_PRISM)}")

	def get_elev(self, row, col):
		
		with rasterio.open(self.SWATGenXPaths.PRISM_dem_path) as src:
			elev_data = src.read(1)
		return elev_data[row, col]

	def extract_from_file(self,f):
		data = f[self.variable][:, self.nsrdb_indexes]
		#self.logger.info(f"NSRDB data shape before getting attr: {data.shape}")	
		scale = f[self.variable].attrs['psm_scale_factor']
		#self.logger.info(f"NSRDB data scale factor: {scale}")
		data = np.divide(data, scale)
		data = np.array(data)  # data shape is (17568, len(NSRDB_index_SW
		#self.logger.info(f"NSRDB data shape after scaling: {data.shape}")
		if self.variable == 'ghi':
			# convert the unit from W/m^2 (30min) to MJ/m^2/day
			data = data.reshape(-1, 48, data.shape[1])  # Reshape to (days, intervals per day, indices)
			data = data * 1800  # Multiply by interval duration in seconds to get energy in J/m²
			daily_data = data.sum(axis=1)  # Sum over intervals to get daily energy
			converter = 1 / 1e6  # Convert J to MJ
			daily_data = daily_data * converter
			#self.logger.info(f"NSRDB data shape after conversion: {daily_data.shape}")	
		elif self.variable in ['wind_speed', 'relative_humidity']:
			daily_data = data.reshape(-1, 48, data.shape[1]).mean(axis=1)
			#self.logger.info(f"NSRDB data shape after conversion: {daily_data.shape}")

		return daily_data

	def fetch_nsrdb(self, year):
		
		file_path = f'/data/SWATGenXApp/GenXAppData/NSRDB/nsrdb_{year}_full_filtered.h5'
		with h5py.File(file_path, mode='r') as f:
			#self.logger.info(f"Extracting {self.variable} for year {year} from CIWRE-BAE server... ")
			daily_data = self.extract_from_file(f)

		#self.logger.info(f"NSRDB data shape for year {year}: {daily_data.shape}")
		return daily_data

	def write_to_file(self,data_all):
		### assert the data_all is not empy
		if len(data_all) < 2:
			self.logger.error(f"Data for {self.variable} is empty")
			return
		
		for nsrdb_idx in self.nsrdb_indexes:
			try:
				row = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].row.values[0]
				col = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].col.values[0]
				#self.logger.info(f"NSRDB_index: {nsrdb_idx}, row: {row}, col: {col}")
				with open(os.path.join(self.output_path, f"r{row}_c{col}.{self.swat_dict[self.variable]}"), 'w') as f:
					f.write(f"NSRDB(/nrel/nsrdb/v3/nsrdb_{self.years[0]}-{self.years[-1]}).INDEX:{nsrdb_idx}\n")
					## write lat lon of the SWAT locations
					lat = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].latitude.values[0]
					lon = self.NSRD_PRISM[self.NSRD_PRISM.NSRDB_index == nsrdb_idx].longitude.values[0]

					f.write("nbyr nstep lat, lon elev\n")
					elev = self.get_elev(row, col)
					f.write(f"{len(self.years)}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")
					date_Range = pd.date_range(start = f"{self.years[0]}-01-01", end = f"{self.years[-1]}-12-31")
					for i in range(data_all.shape[0]):
						j = np.where(self.nsrdb_indexes == nsrdb_idx)[0][0]
						f.write(f"{date_Range[i].year}\t{date_Range[i].dayofyear}\t{data_all[i,j]:.2f}\n")
			except Exception as e:
				self.logger.error(f"Error in writing to file: {e}")	


	def write_cli_file(self):

		cli_name = os.path.join(self.output_path, f"{self.swat_dict[self.variable]}.cli")

		written_rows_cols = set()

		with open(cli_name, 'w') as f:
			f.write("NSRDB(/nrel/nsrdb/v3/nsrdb_2000-2020).INDEX: obtained by h5pyd\n")
			f.write(f"{self.variable} file\n")

			for NSRDB_index, row, col in zip(self.NSRD_PRISM.NSRDB_index.values, self.NSRD_PRISM.row.values, self.NSRD_PRISM.col.values):
				if (row, col) not in written_rows_cols:
					written_rows_cols.add((row, col))
					#self.logger.info(f"NSRDB_index: {NSRDB_index}, row: {row}, col: {col}, {self.variable}")	
					f.write(f"r{row}_c{col}.{self.swat_dict[self.variable]}\n")

	def run(self):
		try:

			self.extract_SWAT_PRISM_locations()
			data_all = []
			for year in self.years:
				daily_var = self.fetch_nsrdb(year)
				if len(data_all) == 0:
					data_all = daily_var
				else:
					data_all = np.concatenate((data_all, daily_var), axis = 0)

			self.logger.info(f"variable {self.variable} has been extracted for SWAT locations {data_all.shape}")
			self.write_to_file(data_all)
			self.write_cli_file()
		except Exception as e:
			self.logger.error(f"Error in NSRDB_contructor: {e}")

def NSRDB_extract(SWATGenXPaths, VPUID, LEVEL, NAME):
	variables = ['ghi','wind_speed','relative_humidity']
	processes = []
	wrapped_extract_variable = partial(nsrdb_contructor_wrapper, SWATGenXPaths = SWATGenXPaths, VPUID = VPUID, LEVEL = LEVEL, NAME = NAME)
	for variable in variables:
		print(f"Extracting {variable} for SWAT PRISM locations")
		p = Process(target = wrapped_extract_variable, args = (variable,))
		processes.append(p)
		p.start()
	for p in processes:
		p.join()
	print("All NSRDB variables have been extracted for SWAT locations")

if __name__ == "__main__":

	LEVEL = 'huc12'
	VPUIDS = get_all_VPUIDs()
	
	VPUID = "0206"
	NAME = "01583570"
	from SWATGenXConfigPars import SWATGenXPaths
	LEVEL = 'huc12'
	username = "vahidr32"	
	SWATGenXPaths = SWATGenXPaths(username="vahidr32", LEVEL = LEVEL, VPUID = VPUID, station_name=NAME)

	NSRDB_extract(SWATGenXPaths, VPUID, LEVEL, NAME)
