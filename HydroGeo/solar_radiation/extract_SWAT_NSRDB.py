### the purpose of this code is to extract the SWAT PRISM locations for NSRDB extractionimport pandas as pd
import numpy as np
import pandas as pd
import geopandas as gpd
import h5pyd
import os
from functools import partial
from multiprocessing import Process

def extract_SWAT_PRISM_locations(swat_prism_shape_path):
	PRISM_SWAT = gpd.read_file(swat_prism_shape_path)
	PRISM_SWAT['ROWCOL'] = PRISM_SWAT['row'].astype(str) + PRISM_SWAT['col'].astype(str)
	NSRD_PRISM = pd.read_pickle("/data/MyDataBase/SWATGenXAppData/NSRDB/PRISM_NSRDB_CONUS.pkl")
	NSRD_PRISM['ROWCOL'] = NSRD_PRISM['row'].astype(str) + NSRD_PRISM['col'].astype(str)
	NSRDB_SWAT = pd.merge(NSRD_PRISM.drop(columns=['row','col','geometry']), PRISM_SWAT, on = "ROWCOL", how = "inner")
	NSRDB_SWAT['NSRDB_index'] = NSRDB_SWAT['NSRDB_index'].astype(int)
	print(f"NSRDB_SWAT colums: {NSRDB_SWAT.columns.values}")
	return np.unique(NSRDB_SWAT.sort_values(by ='NSRDB_index').NSRDB_index.values), NSRD_PRISM[NSRD_PRISM.NSRDB_index.isin(np.unique(NSRDB_SWAT.NSRDB_index.values))]


def fetch_nsrdb(year, variable, NSRDB_index_SWAT):
	file_path = f'/nrel/nsrdb/v3/nsrdb_{year}.h5'
	print(f"Extracting {variable} for year {year}... ")
	with h5pyd.File(file_path, mode='r') as f:
		data = f[variable][:,NSRDB_index_SWAT]
		print(f"NSRDB data shape before getting attr: {data.shape}")
		scale = f[variable].attrs['psm_scale_factor']
		print("rescaling data....")
		data = np.divide(data, scale)
		data = np.array(data) #  data shape is (17568, len(NSRDB_index_SWAT))
		print(f"NSRDB data shape after scaling: {data.shape}")
		if variable == 'ghi':
			# convert the unit from Wh/m^2 (30min) to MJ/m^2/day
			daily_data = data.reshape(-1, 48, data.shape[1]).sum(axis = 1) ## resample from 30 minutes to daily
			converter = 1/1e6*86400
			daily_data = daily_data * converter
		elif variable in ['wind_speed','relative_humidity']:
			daily_data = data.reshape(-1, 48, data.shape[1]).mean(axis = 1)

	return daily_data

def write_to_file(variable, NSRDB_index_SWAT, years, data_all, swat_prism_path , NSRD_PRISM):
	swat_dict = {'ghi':'slr',
	'wind_speed':'wnd',
	'relative_humidity':'hmd',
	}
	for NSRDB_index in NSRDB_index_SWAT:
		row = NSRD_PRISM[NSRD_PRISM.NSRDB_index == NSRDB_index].row.values[0]
		col = NSRD_PRISM[NSRD_PRISM.NSRDB_index == NSRDB_index].col.values[0]
		print(f"NSRDB_index: {NSRDB_index}, row: {row}, col: {col}")
		with open(os.path.join(swat_prism_path, f"r{row}_c{col}.{swat_dict[variable]}"), 'w') as f:
			f.write(f"NSRDB(/nrel/nsrdb/v3/nsrdb_{years[0]}-{years[-1]}).INDEX:{NSRDB_index}, obtained by h5pyd\n")
			## write lat lon of the SWAT locations
			lat = NSRD_PRISM[NSRD_PRISM.NSRDB_index == NSRDB_index].latitude.values[0]
			lon = NSRD_PRISM[NSRD_PRISM.NSRDB_index == NSRDB_index].longitude.values[0]
			#print(f"{variable},NSRDB_index: {NSRDB_index}, lat: {lat}, lon: {lon}")
			## format: year, day, value
			f.write("nbyr nstep lat, lon elev\n")
			elev = 200
			f.write(f"{len(years)}\t0\t{lat:.2f}\t{lon:.2f}\t{elev:.2f}\n")
			#f.write(f"year\tday\t{variable}\n")
			date_Range = pd.date_range(start = f"{years[0]}-01-01", end = f"{years[-1]}-12-31")
			for i in range(data_all.shape[0]):
				j = np.where(NSRDB_index_SWAT == NSRDB_index)[0][0]
				f.write(f"{date_Range[i].year}\t{date_Range[i].dayofyear}\t{data_all[i,j]:.2f}\n")


def write_cli_file(swat_prism_path, NSRD_PRISM, variable):
	swat_dict = {'ghi':'slr',
	'wind_speed':'wnd',
	'relative_humidity':'hmd',
	}
	cli_name = os.path.join(swat_prism_path, f"{swat_dict[variable]}.cli")
	with open(cli_name,'w') as f:
		f.write("NSRDB(/nrel/nsrdb/v3/nsrdb_2000-2020).INDEX: obtained by h5pyd\n")
		f.write(f"{variable} file\n")
		for NSRDB_index, row, col in zip(NSRD_PRISM.NSRDB_index.values, NSRD_PRISM.row.values, NSRD_PRISM.col.values):
			NSRD_PRISM = NSRD_PRISM[NSRD_PRISM.NSRDB_index == NSRDB_index]
			print(f"NSRDB_index: {NSRDB_index}, row: {row}, col: {col},{variable}")
			f.write(f"r{row}_c{col}.{swat_dict[variable]}\n")

def extract_SWAT_PRISM_variable(variable, NSRDB_index_SWAT, years, swat_prism_path, NSRD_PRISM):
	data_all = []
	for year in years:
		print(f"Extracting {variable} for year {year}")
		daily_var = fetch_nsrdb(year,variable, NSRDB_index_SWAT)
		if len(data_all) == 0:
			data_all = daily_var
		else:
			data_all = np.concatenate((data_all, daily_var), axis = 0)
	print(f"variable {variable} has been extracted for SWAT locations", data_all.shape)
	write_to_file(variable, NSRDB_index_SWAT, years, data_all, swat_prism_path, NSRD_PRISM)
	write_cli_file(swat_prism_path, NSRD_PRISM, variable)

def NSRDB_extract(VPUID,NAME,LEVEL):
	years = range(2000, 2021)
	DIC = "/data/MyDataBase/SWATGenXAppData/"
	variables = ['ghi','wind_speed','relative_humidity']

	swat_prism_path = os.path.join(DIC, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/PRISM/")
	swat_prism_shape_path = os.path.join(swat_prism_path,"PRISM_grid.shp")
	NSRDB_index_SWAT,NSRD_PRISM = extract_SWAT_PRISM_locations(swat_prism_shape_path)
	print(f"NSRDB_index_SWAT columns: {NSRD_PRISM.columns.values}")
	processes = []
	wrapped_extract_variable = partial(extract_SWAT_PRISM_variable, NSRDB_index_SWAT = NSRDB_index_SWAT, years = years, swat_prism_path = swat_prism_path, NSRD_PRISM = NSRD_PRISM)
	for variable in variables:
		print(f"Extracting {variable} for SWAT PRISM locations")
		p = Process(target = wrapped_extract_variable, args = (variable,))
		processes.append(p)
		p.start()
	for p in processes:
		p.join()
	print("All NSRDB variables have been extracted for SWAT locations")

if __name__ == "__main__":
	VPUID = "0202"
	NAME = '01333000'
	LEVEL = 'huc12'
	NSRDB_extract(VPUID,NAME,LEVEL)