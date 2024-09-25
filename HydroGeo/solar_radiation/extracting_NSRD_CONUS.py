import h5py
import pandas as pd
import numpy as np
import geopandas as gpd 
import tqdm
import os 
from datetime import datetime
import time
import concurrent.futures
from functools import partial
def fetch_CONUS_NSRDB_index():
    PRISM_path = fr"D:\MyDataBase\PRISM\prism_4km_mesh\prism_4km_mesh.pkl"
    PRISM_coordinates = gpd.GeoDataFrame(pd.read_pickle(PRISM_path), geometry='geometry', crs='EPSG:4326')
    extent_of_PRISM = PRISM_coordinates.total_bounds

    file_path = 'E:/NSRDB/nsrdb_2017_full.h5'
    CONUS_coordinates_index = []

    if not os.path.exists('E:/NSRDB/CONUS_coordinates_index.csv'):
        with h5py.File(file_path, 'r') as f:
            coordinates = f['coordinates']
            for i in range(len(coordinates)):
                coordinate = coordinates[i]
                if extent_of_PRISM[0] < coordinate[1] < extent_of_PRISM[2] and extent_of_PRISM[1] < coordinate[0] < extent_of_PRISM[3]:
                    CONUS_coordinates_index.append([i, coordinate[0], coordinate[1]])
            pd.DataFrame(CONUS_coordinates_index, columns=['NSRDB_index', 'latitude', 'longitude']).to_csv('E:/NSRDB/CONUS_coordinates_index.csv', index=False)

    if not os.path.exists('E:/NSRDB/CONUS_coordinates_index.pkl'):
        CONUS_coordinates_index = pd.read_csv('E:/NSRDB/CONUS_coordinates_index.csv')
        CONUS_coordinates_index = gpd.GeoDataFrame(CONUS_coordinates_index, geometry=gpd.points_from_xy(CONUS_coordinates_index.longitude, CONUS_coordinates_index.latitude), crs='EPSG:4326')
        CONUS_coordinates_index.to_file('E:/NSRDB/CONUS_coordinates_index')
        CONUS_coordinates_index.to_pickle('E:/NSRDB/CONUS_coordinates_index.pkl')

    return pd.read_pickle(
        'E:/NSRDB/CONUS_coordinates_index.pkl'
    ).NSRDB_index.values


def write_chunk_to_file(chunk, new_file_path, parameter):
    with h5py.File(new_file_path, 'a') as f:
        parameter_data = f[parameter]
        time_index = pd.to_datetime(f['time_index'][:].astype(str))
        for NSRDB_index in chunk:
            parameter_data = parameter_data[:, NSRDB_index]
            dataframe = pd.DataFrame({'time_index': time_index, parameter: parameter_data})
            dataframe.set_index('time_index', inplace=True)
            daily_sum = dataframe.resample('D').sum()[parameter].values
            f.create_dataset(f'NSRDB_index_{NSRDB_index}', data=daily_sum)
            
def initialize_file(new_file_path, parameter, CONUS_NSRDB_index, psm_units, psm_scale_factor, description, year):  
    with h5py.File(new_file_path, 'w') as f:
        print(f'Creating file: {new_file_path}')
        print(f'lenght of CONUS_NSRDB_index: {len(CONUS_NSRDB_index)}')
        
        time.sleep(5)
        number_of_years = 366 if year == ['2000', '2004', '2008', '2012', '2016', '2020'] else 365
        
        parameter_data = f.create_dataset(parameter, (number_of_years, len(CONUS_NSRDB_index)), dtype='f4')
        parameter_data.attrs['psm_units'] = psm_units
        parameter_data.attrs['psm_scale_factor'] = psm_scale_factor


def split_into_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def main(year, parameter, CONUS_NSRDB_index):
    print(f'len(CONUS_NSRDB_index): {len(CONUS_NSRDB_index)}')  
    file_path = f'E:/NSRDB/nsrdb_{year}_full.h5'
    
    new_file_path = f'E:/NSRDB/nsrdb_{year}_daily_{parameter}.h5'
    
    with h5py.File(file_path, 'r') as f:
        psm_scale_factor = f[parameter].attrs['psm_scale_factor']
        psm_units = f[parameter].attrs['psm_units']
    if not os.path.exists(new_file_path):
        if parameter in ['ghi']:
            description = 'resampled daily sum'
        else:
            description = 'resampled daily mean'
        initialize_file(new_file_path, parameter, CONUS_NSRDB_index, psm_units=psm_units, psm_scale_factor=psm_scale_factor, description=description, year=year)
        print(f'Initialized file: {new_file_path}')
    
    with h5py.File(file_path, 'r') as f:
        time_index = pd.to_datetime(f['time_index'][:].astype(str))
        with h5py.File(new_file_path, 'a') as new_f:
            for NSRDB_index in tqdm.tqdm(CONUS_NSRDB_index):
                
                parameter_data = f[parameter][:, NSRDB_index]
                dataframe = pd.DataFrame({'time_index': time_index, parameter: parameter_data})
                dataframe.set_index('time_index', inplace=True)
                if parameter in ['ghi']:
                    daily_sum = dataframe.resample('D').sum()[parameter].values
                else:
                    daily_sum = dataframe.resample('D').mean()[parameter].values
                new_f.create_dataset(f'NSRDB_index_{NSRDB_index}', data=daily_sum)


if __name__ == '__main__':
    CONUS_NSRDB_index = fetch_CONUS_NSRDB_index()
    parameters = ['wind_speed', 'relative_humidity', 'ghi']
    processes = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for year in ['2017', '2016', '2018']:
            for parameter in parameters:
                wrapped_processes = partial(main, year=year, parameter=parameter, CONUS_NSRDB_index=CONUS_NSRDB_index)
                process = executor.submit(wrapped_processes)
                processes.append(process)
    for process in concurrent.futures.as_completed(processes):
        process.result()
