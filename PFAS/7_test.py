import h5py
import numpy as np
with h5py.File("/data/MyDataBase/SWATGenXAppData/codes/PFAS/PFAS_sw_30m.h5", 'r') as f:
	df = f['/Max/PFOS']
	### read mean values. novalue is -999
	df = df[...]
	# Replace no-data values with NaN
	df[df == -999] = np.nan	
	print(df.shape)
	print(f"mean: {np.nanmean(df)}")
	print(f"min: {np.nanmin(df)}")
	print(f"max: {np.nanmax(df)}")
	print(f"std: {np.nanstd(df)}")
	print(f"number of values without nan: {np.count_nonzero(~np.isnan(df))}")
	print(f"number of nan values: {np.count_nonzero(np.isnan(df))}")