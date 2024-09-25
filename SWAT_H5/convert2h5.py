import itertools
import numpy as np
import pandas as pd
import os
import h5py
from unit_global import get_channel_unit, get_hrus_unit

def write_hru_metadata(DIC, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO):
	def create_hru_datasets(f, df, column_name, dataset_name):
		f.create_dataset(f"metadata/hru/{dataset_name}", data=df[column_name].values)

	""" Write HRU metadata to the h5 file"""
	hru_data = os.path.join(DIC, f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/{SCENARIO}/hru-data.hru")
	output_h5_path =  os.path.join(DIC,'codes/SWAT_H5/',"SWAT_OUTPUT.h5")    
	df = pd.read_csv(hru_data, sep='\s+', skiprows =1)
	for col in df.columns:
		df[col] = df[col].astype('str')
	with h5py.File(output_h5_path, 'a') as f:
		create_hru_datasets(f, df, 'name', 'name')
		create_hru_datasets(f, df, 'topo', 'topo')
		create_hru_datasets(f, df, 'hydro', 'hydro')
		create_hru_datasets(f, df, 'soil', 'soil')
		create_hru_datasets(f, df, 'lu_mgt', 'lu_mgt')
		create_hru_datasets(f, df, 'soil_plant_init', 'soil_plant_init')
		create_hru_datasets(f, df, 'surf_stor', 'surf_stor')
		create_hru_datasets(f, df, 'snow', 'snow')
		create_hru_datasets(f, df, 'field', 'field')

def read_SWAT_data(swat_output_file):
	# Read the header to use as column names, skipping the lines
	with open(swat_output_file, 'r') as file:
		header = file.readlines()[1].strip().split()  # Second line is the actual header
	header = [f'null{i+1}' if col == 'null' else col for i, col in enumerate(header)]
	df = pd.read_csv(swat_output_file, sep='\s+', skiprows=3, names=header)
	df.drop(columns=[col for col in df.columns if 'null' in col], inplace=True)
	components = len(df['name'].unique())
	names = df['name'].unique()
	months = df['mon'].unique()
	years = df['yr'].unique()
	jdays = df['jday'].unique()
	df['id'] = pd.factorize(df['name'])[0]
	ts_length = len(df) // components
	return df.sort_values(by=['name', 'yr', 'mon']), components, ts_length, months, years, names
	

def convert2h5(DIC, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, TIME_STEPS, FILE_NAMES):
	# sourcery skip: low-code-quality

	output_h5_path = os.path.join(DIC,'codes/SWAT_H5/',"SWAT_OUTPUT.h5")
	
	if os.path.exists(output_h5_path):
		os.remove(output_h5_path)
	
	# Rewrite and refactor. Write all output files into one h5 with different hierarchy
	for FILE_NAME, TIME_STEP in itertools.product(FILE_NAMES, TIME_STEPS):
				
		print(f"###### Working on {FILE_NAME} at {TIME_STEP} time step ######")
		
		swat_output_file = os.path.join(DIC, fr"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}/Scenarios/{SCENARIO}/{FILE_NAME}_{TIME_STEP}.txt")
		
		if not os.path.exists(swat_output_file):
			continue
		
		df, components, ts_length, months, years, names = read_SWAT_data(swat_output_file)
		
		open_method = 'a' if os.path.exists(output_h5_path) else 'w'
		
		with h5py.File(output_h5_path, open_method) as f:
			for i, parameter in enumerate(df.columns):
				if parameter in ['jday', 'mon', 'day', 'yr', 'unit', 'gis_id']:
					continue
				parameter_hrus = df[parameter].values.reshape(components, ts_length).T
				assert np.all(df['id'].values.reshape(components, ts_length).T == np.arange(components)[None, :])
				print(f"Writing {parameter} to h5 file as {FILE_NAME}/{TIME_STEP}/{parameter} with shape {parameter_hrus.shape}")
				f.create_dataset(f"{FILE_NAME}/{TIME_STEP}/{parameter}", data=parameter_hrus)
				
				f[f"{FILE_NAME}/{TIME_STEP}/{parameter}"].attrs['unit'] = get_channel_unit(parameter) if FILE_NAME == 'channel_sd' else get_hrus_unit(parameter)


			if "metadata/months" not in f:
				f.create_dataset("metadata/months", data=months)
			if "metadata/years" not in f:
				f.create_dataset("metadata/years", data=years)


# Define the path to the SWAT HRU monthly data file
if __name__ == "__main__":
	NAME = "40500010102"
	VPUID = "0405"
	LEVEL = "huc12"
	MODEL_NAME = "SWAT_gwflow_MODEL"
	SCENARIO = "Scenario_verification_stage_1"
	DIC = "/data/MyDataBase/SWATGenXAppData/"
	TIME_STEPS = ["yr", "mon"]
	FILE_NAMES = ["channel_sd", 'hru_wb']
	convert2h5(DIC, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO, TIME_STEPS, FILE_NAMES)
	write_hru_metadata(DIC, VPUID, LEVEL, NAME, MODEL_NAME, SCENARIO)
