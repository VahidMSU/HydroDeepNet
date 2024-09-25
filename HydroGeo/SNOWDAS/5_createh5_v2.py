import h5py
import numpy as np
from datetime import datetime
import os
import time
import itertools
import rasterio
import matplotlib.pyplot as plt
from matplotlib import animation
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
    return data

def check_calendar(years, months, days):
    for year in years:
        for month, day in itertools.product(months, days):
            try:
                datetime(year, month, day)
            except ValueError:
                return False
    return True

def create_group(h5_file, group_name):
    if group_name in h5_file:
        return h5_file[group_name]
    return h5_file.create_group(group_name)

def get_reference_dimention():
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    data = read_raster(reference_raster)
    return data.shape

def read_SNODAS_rasters(h5_file_path, group_name, var, year, no_data_value=55537):
    months = np.arange(1, 13)
    days = np.arange(1, 32)
    all_days_data = []
    snowdas_path = "/data/MyDataBase/SWATGenXAppData/snow/snow/michigan/"

    common_shape = get_reference_dimention()

    for month, day in itertools.product(months, days):
        path = os.path.join(snowdas_path, str(year), str(month), str(day))
        if check_calendar([year], [month], [day]) is False:
            continue
        if not os.path.exists(path):
            #print(f"Path {path} does not exist")
            data = np.ones(common_shape) * no_data_value
            all_days_data.append(data)
            continue
        files = os.listdir(path)
        tif_files = [f for f in files if f.endswith(".tif") and "_resample" in f and var in f]
        files = [os.path.join(path, f) for f in tif_files]
        len_files = len(files)
        if len_files == 0:
            #print(f"No files found in {path}")
            data = np.ones(common_shape) * no_data_value
            all_days_data.append(data)
            continue
        for file in files:
            data = read_raster(file)
            data = data[:common_shape[0], :common_shape[1]]
            all_days_data.append(data)

    return np.stack(all_days_data, axis=0)

def create_h5_file(h5_file_path, group_name, variable_name, all_days_data):
    with h5py.File(h5_file_path, "a") as h5_file:
        group = create_group(h5_file, group_name)
        ## use unit16 to save space
        dataset = group.create_dataset(variable_name, data=np.array(all_days_data), dtype='uint16')
        converters, units = get_convertor_unit(variable_name)
        dataset.attrs['converters'] = converters
        dataset.attrs['units'] = units

def read_h5_file(h5_file_path, group_name, variable_name):
    with h5py.File(h5_file_path, "r") as h5_file:
        group = h5_file[group_name]
        data = group[variable_name][:]
    return data
def video_data(data, name, no_data_value=55537):
    ### create video of the data, each 2d array from the 3d array is a frame
    ### data: 3d array
    fig, ax = plt.subplots()
    data = np.where(data == no_data_value, np.nan, data)
    def update_frame(i):
        ax.clear()
        im = ax.imshow(data[i], animated=True)
        return [im]

    ani = animation.FuncAnimation(fig, update_frame, frames=range(data.shape[0]), interval=50, blit=True, repeat_delay=1000)
    ani.save(f'{name}.gif', writer='pillow')
    plt.close(fig)
    print(f"Video of {name} data saved.")


def get_convertor_unit(variable_name):
    variable_names = ['average_temperature',
                    'melt_rate',
                    'snow_layer_thickness',
                    'snow_water_equivalent',
                    'snowpack_sublimation_rate',
                    'non_snow_accumulation',
                    'snow_accumulation']

    converters = [1, 1 / 100, 1, 1, 1 / 100, 1 / 10, 1 / 10]

    units = ['Kelvin',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'kg/sqm',
            'kg/sqm']


    return converters[variable_names.index(variable_name)], units[variable_names.index(variable_name)]

def initiate_h5_file(h5_file_path):
    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
        print(f"File {h5_file_path} has been deleted.")


def main():
    years = np.arange(2004, 2024)
    h5_file_path = "/data/MyDataBase/SWATGenXAppData/snow/snow/michigan/SNODAS.h5"
    variable_names = [
                'melt_rate',
                'snow_layer_thickness',
                'snow_water_equivalent',
                'snowpack_sublimation_rate',
                'snow_accumulation']

    initiate_h5_file(h5_file_path)
    for year in years:
        group_name = f"/250m/{year}"
        for variable_name in variable_names:
            print(f"Processing {variable_name}", "year", year)
            all_days_data = read_SNODAS_rasters(h5_file_path, group_name, variable_name, year)
            create_h5_file(h5_file_path, group_name, variable_name, all_days_data)
            data = read_h5_file(h5_file_path, group_name, variable_name)
            #video_data(data, variable_name)
            print(data.shape)

if __name__ == "__main__":
    main()
