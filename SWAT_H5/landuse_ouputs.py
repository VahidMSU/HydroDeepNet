import h5py
import numpy as np
import pandas as pd
import os 
import numpy as np
from unit_global import get_channel_unit, get_hrus_unit, list_of_variables

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_metadata(file_path, time_step):
	""" Read metadata from the h5 file"""
	with h5py.File(file_path, 'r') as f:
		landuse = f['metadata/hru/lu_mgt'][:].astype(str)
		soil = f['metadata/hru/soil'][:].astype(str)
		names = f['metadata/hru/name'][:].astype(str)
		years = f['metadata/years'][:]
		months = f['metadata/months'][:]
		start_year = years[0]
		end_year = years[-1]
		start_month = months[0]
		end_month = months[-1]
		timerange = pd.date_range(start=f'{start_year}-{start_month:02d}', end=f'{end_year}-{end_month:02d}', freq="M" if time_step == 'mon' else "Y") 
		metadataframe = pd.DataFrame({
			'name': names,
			'soil': soil,
			'landuse': landuse
		})
		return metadataframe, timerange

def process_mvar_data(file_path, hru_vars):
	with h5py.File(file_path, 'r') as f:
		for time_step in ['mon', 'yr']:
			metadataframe, timerange = read_metadata("/data/MyDataBase/SWATGenXAppData/codes/swat_h5/SWAT_OUTPUT.h5", time_step)

			for var_name in hru_vars:
				mvar = f[f'hru_wb/{time_step}/{var_name}'][:]
				names_mvar = f[f'hru_wb/{time_step}/name'][0,:].astype(str)
				mvar_df = pd.DataFrame(mvar, columns=names_mvar).T
				mvar_df.columns = [f'{str(i)}mon' for i in range(1, mvar_df.shape[1]+1)]
				mvar_df['name'] = names_mvar
				mvar_df = pd.merge(mvar_df, metadataframe, on='name')
				mvar_df = mvar_df.drop(columns=['soil','name']).groupby('landuse').mean().reset_index()
				mvar_df['landuse'] = [x.split('_')[0] for x in mvar_df['landuse']]
				mvar_df['landuse'] = [x.upper() for x in mvar_df['landuse']]
				df = mvar_df.set_index('landuse').T
				df = pd.DataFrame(df.to_records())
				df = df.drop(df.index[-1])
				plot_landuse_data(df, var_name, time_step, timerange)

def plot_landuse_data(df, var_name, time_step, timerange):
	fig, ax = plt.subplots(figsize=(10, 6))
	# use viridis colormap
	colors = cm.viridis(np.linspace(0, 1, len(df.columns[1:])))
	for landuse, color in zip(df.columns[1:], colors):
		ax.plot(timerange, df[landuse], label=landuse, color=color)
	ax.set_title(f'{var_name} for each landuse')
	ax.set_xlabel("Months" if time_step == 'mon' else "Years")
	ax.set_ylabel(get_hrus_unit(var_name))
	ax.legend(labels=df.columns.values[1:], loc='upper right')
	plt.savefig(f"/data/MyDataBase/SWATGenXAppData/codes/swat_h5/figs/landuse_{var_name}_{time_step}.png", dpi=300)

if __name__ == '__main__':
	dataset_name = 'hru_wb'
	hru_vars = list_of_variables(dataset_name)
	process_mvar_data("/data/MyDataBase/SWATGenXAppData/codes/swat_h5/SWAT_OUTPUT.h5", hru_vars)
