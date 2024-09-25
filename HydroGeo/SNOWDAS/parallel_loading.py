import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import geopandas as gpd
import concurrent.futures
import itertools
import netCDF4 as nc

def load_variable(file_path, var_name):
    with nc.Dataset(file_path) as dataset:
        return dataset.variables[var_name][:]

def loading_dataset(BASE_PATH):
    # Load latitudes and longitudes
    SNODAS_stations = gpd.read_file(os.path.join(BASE_PATH, 'snow/SNODAS_locations.shp'))
    latitudes = np.loadtxt(os.path.join(BASE_PATH, 'snow/SNODAS_latitudes_michigan.txt'))
    longitudes = np.loadtxt(os.path.join(BASE_PATH, 'snow/SNODAS_longitudes_michigan.txt'))

    # Define file paths and variable names
    datasets = {
        'average_temperature': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_average_temperature_constrained_2004_2023.nc'),
        'blowing_snow_sublimation_rate': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_blowing_snow_sublimation_rate_constrained_2004_2023.nc'),
        'melt_rate': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_melt_rate_constrained_2004_2023.nc'),
        'snow_layer_thickness': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_snow_layer_thickness_constrained_2004_2023.nc'),
        'snow_water_equivalent': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_snow_water_equivalent_constrained_2004_2023.nc'),
        'snowpack_sublimation_rate': os.path.join(BASE_PATH, 'snow/SNODAS_Modeled_snowpack_sublimation_rate_constrained_2004_2023.nc'),
        'non_snow_accumulation': os.path.join(BASE_PATH, 'snow/SNODAS_Non_snow_accumulation_constrained_2004_2023.nc'),
        'snow_accumulation': os.path.join(BASE_PATH, 'snow/SNODAS_Snow_accumulation_constrained_2004_2023.nc')
    }

    # Load data from NetCDF files using parallel processing
    variables = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {key: executor.submit(load_variable, path, 'value') for key, path in datasets.items()}
        for key, future in futures.items():
            variables[key] = future.result()

    return variables, longitudes, latitudes, SNODAS_stations, datasets



if __name__ == '__main__':
    BASE_PATH = 'D:/MyDataBase'
    variables, longitudes, latitudes , SNODAS_stations, datasets = loading_dataset(BASE_PATH)


