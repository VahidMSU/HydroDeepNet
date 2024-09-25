import os
from venv import create
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_huc8_ranges(RESOLUTION, huc8_select):

    database_path = '/data/MyDataBase/HydroGeoDataset_ML_250.h5'

    with h5py.File(database_path, 'r') as f:
        huc8 = np.array(f['HUC8_250m'][:])

    rows, cols = np.where(huc8 == int(huc8_select))
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    return row_min, row_max, col_min, col_max

def video_data(data, name):
    ### create video of the data, each 2d array from the 3d array is a frame
    ### data: 3d array
    fig, ax = plt.subplots()

    def update_frame(i):
        ax.clear()
        im = ax.imshow(data[i], animated=True)
        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=range(data.shape[0]), interval=50, blit=True, repeat_delay=1000)

    os.makedirs('input_videos', exist_ok=True)
    ani.save(f'input_videos/{name}.gif', writer='pillow')
    plt.close(fig)
    print(f"Video of {name} data saved.")
def extract_snowdas_data(self, snowdas_h5_path, snowdas_var, huc8_select):

    """ parameter to extract the data from the SNODAS dataset.
    Parameters:
        'average_temperature', 'melt_rate', 'non_snow_accumulation', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate'
    Geographic extent:
        Michigan LP

    Example:
    extract_snowdas_data(snowdas_h5_path, 'snowdas_var', 44, 47, -87, -84)
    """
    with h5py.File(snowdas_h5_path, "r") as h5_file:
        print(h5_file['250m/2004'].keys())

        var = h5_file[f"250m/2004/{snowdas_var}"][:]
        unit = h5_file[f"250m/2004/{snowdas_var}"].attrs['units']
        print(f"Size of the SNOWDAS data: {var.shape}")
        convertor = h5_file[f"250m/2004/{snowdas_var}"].attrs['converters']
        print(f"Convertor: {convertor}")
        var = np.where(var == 55537, np.nan, var*convertor)
        if huc8_select:
            
            min_x, max_x, min_y, max_y  = get_huc8_ranges(250, huc8_select)
            var = var[:, min_x:max_x, min_y:max_y]
            

        print(f"Size of the SNOWDAS data after cropping: {var.shape}")
        video_data(var, f"{snowdas_var}_{unit}")

if __name__ == "__main__":
    snowdas_h5_path = "/data/MyDataBase/SNODAS.h5"
    huc8_select = "4060105"
    extract_snowdas_data(snowdas_h5_path, 'melt_rate', huc8_select)
