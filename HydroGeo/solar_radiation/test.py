import h5pyd
year = 2014
file_path = f'/nrel/nsrdb/v3/nsrdb_{year}.h5'
with h5pyd.File(file_path, mode='r') as f:
	data = f['wind_speed'][:,4000].min()
	print(f['wind_speed'].attrs['psm_units'])
	print(f['wind_speed'].attrs['psm_scale_factor'])
	print(data)