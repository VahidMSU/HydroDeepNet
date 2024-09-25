import h5py    
import numpy as np
import os

### 
import matplotlib.pyplot as plt

def visualize_gwflow_data(path, fig_path):
	with h5py.File(path, 'r') as f:
		print(f.keys())
		for component in f['gwflow'].keys():
			for var in f['gwflow'][component].keys():
				for year in f['gwflow'][component][var].keys():
					
					array = f['gwflow'][component][var][year][:, :]
					array = array.astype(float)  # Convert image data to float type
					### replace 0 with nan
					array[array == 0] = np.nan
					
					plt.imshow(array)
					plt.title(f"{component} {var} {year}")
					plt.colorbar()
					plt.savefig(os.path.join(fig_path, f"{component}_{var}_{year}.png"), dpi=300)
					plt.close()
from watershed_outputs import clean_dir
if __name__ == "__main__":
	path = "/data/MyDataBase/SWATGenXAppData/codes/SWAT_H5/SWAT_OUTPUT.h5"
	fig_path = "/data/MyDataBase/SWATGenXAppData/codes/SWAT_H5/figs"
	clean_dir(fig_path)
	visualize_gwflow_data(path, fig_path)