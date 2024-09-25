
from GeoNet.import_data import get_huc8_ranges
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
def get_mask(config):
    with h5py.File(config['database_path'], 'r') as f:
        DEM_ = f[f'BaseRaster_{config["RESOLUTION"]}m'][:]
        ### get huc8 ranges
        row_min, row_max, col_min, col_max = get_huc8_ranges(config)
        DEM_ = DEM_[row_min:row_max, col_min:col_max]
        return DEM_ != -999

def plot_loss(losses, config):
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epoch')
    plt.grid(alpha=0.5, linestyle='--', color='black')
    plt.savefig(os.path.join(config['fig_path'], f'FFR_{config["target_array"]}_loss_over_epoch.png'), dpi=300)
    plt.close()

def plot_grid(config, array, name):
    mask = get_mask(config)
    array[~mask] = np.nan
    # first save the array
    
    with h5py.File(f'/data/MyDataBase/out/{name}.h5', 'w') as f:
        f.create_dataset(name, data=array)
    vmin, vmax = np.nanpercentile(array, [2.5, 97.5])
    plt.imshow(array, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(name)
    plt.savefig(os.path.join(config['fig_path'], f'{name}.png'), dpi=300)
    plt.close()
