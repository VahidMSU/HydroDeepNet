import h5py
import numpy as np
from rasterio.plot import show
import os

import matplotlib.pyplot as plt

path = "Z:/HydroGeoDataset_ML_250.h5"
fig_path = "/data/MyDataBase/SWATGenXAppData/codes/mldata_creation_h5_rasters/figs/"

if not os.path.exists(fig_path):
	os.makedirs(fig_path)
else:
	for fig in os.listdir(fig_path):
		os.remove(os.path.join(fig_path, fig))

# read SWL
with h5py.File(path, 'r') as f:
	print(f.keys())
	for key in f.keys():
		try:
			print(key)
			data = f[key][:]

			# Handle NaN values
			data[np.isnan(data)] = 0

			# Convert data to integers
			data = data.astype(np.int32)
			data = np.where(data < 0, np.nan, data)
			plt.imshow(data, cmap='viridis', vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))
			plt.title(key)
			plt.colorbar()
			plt.savefig(os.path.join(fig_path, f'{key}.png'))
			plt.close()
		except Exception as e:
			print(f"Error: {e}")
