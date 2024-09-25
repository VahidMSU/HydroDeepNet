import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from GeoCNN.viz import plot_gssurgo_grid
import sqlite3

def generate_ssurgo_csv(swatplus_soils):
    ### open and get ssurgo layer and ssurgo data
    conn = sqlite3.connect(path)

    query = "SELECT * FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    print(tables)

    ssurgo_layer = pd.read_sql_query("SELECT * FROM ssurgo_layer", conn)

    ssurgo_data = pd.read_sql_query("SELECT * FROM ssurgo", conn)

    print(ssurgo_layer.iloc[:, 0:10])

    print(ssurgo_data.iloc[:, 0:10])

    ## merge dataframe ssurgo_layer and ssurgo_data, left on soil_id and right on id

    ssurgo = pd.merge(ssurgo_layer, ssurgo_data, left_on='soil_id', right_on='id')

    print(ssurgo.iloc[:, 0:10])

    ### save the merged dataframe to a csv file

    ssurgo.to_csv('database/ssurgo.csv', index=False)

def load_gssurgo_grid(gssurgo_raster_path):
    import rasterio
    with rasterio.open(gssurgo_raster_path) as src:
        gssurgo_grid = src.read(1)
        print(f"Shape of gssurgo_grid: {gssurgo_grid.shape}")
        unique_muid = np.unique(gssurgo_grid)
        ### get the no value
        no_value = src.nodata
        print(f"Unique values in gssurgo_grid: {unique_muid}")
        unique_muid = unique_muid[unique_muid != no_value]
    return gssurgo_grid, unique_muid, no_value

def add_to_hdf5(column, _gssurgo_grid, database_path):
    with h5py.File(database_path, 'a') as f:
        ## write columns to the hdf5 file under gssurgo group with float32
        def write_to_hdf5(column, data):
            ### if not already exists
            if f'gssurgo/{column}' in f:
                ## remove the existing dataset
                del f[f'gssurgo/{column}']

            ## create a new dataset
            ## replace nan with -999
            data = np.where(np.isnan(data), -999, data)
            f.create_dataset(f'gssurgo/{column}', data=data, dtype='float32')

        write_to_hdf5(column, _gssurgo_grid)

def create_h5py_file(database_path):
    import os 
    if not os.path.exists(database_path):
        with h5py.File(database_path, 'w') as f:
            ## create a group called gssurgo
            f.create_group('gssurgo')

def replace_nan_with_no_value(no_value, gssurgo_grid, unique_muid, column, grouped):
    _gssurgo_grid = gssurgo_grid.astype(float)  # Convert to float to handle NaN values
    _gssurgo_grid = np.nan * np.ones_like(_gssurgo_grid)
    for muid in unique_muid:
        
        # Replace gssurgo_grid with the mean value for the current column
        try:
            mean_value = grouped.loc[muid, column]
            _gssurgo_grid = np.where(gssurgo_grid == muid, mean_value, _gssurgo_grid)
        except KeyError:
            print(f"muid {muid} not in grouped data")
            
            continue
    # Replace -999 with NaN for no values
    _gssurgo_grid = np.where(_gssurgo_grid == -999, np.nan, _gssurgo_grid)
    _gssurgo_grid = np.where(_gssurgo_grid == no_value, np.nan, _gssurgo_grid)
    return _gssurgo_grid

def process_soil_data(_complete_soil_data, soil_property, gssurgo_raster_path, database_path):
    complete_soil_data = _complete_soil_data.copy()
    print(complete_soil_data.columns)
    complete_soil_data = complete_soil_data[['muid', soil_property]]
    ## if soil_property datatype is not float, use factorize to convert to int
    assert "muid" in complete_soil_data.columns, "muid column not found in complete_soil_data"
    assert soil_property in complete_soil_data.columns, "soil_k column not found in complete_soil_data"
    ## drop nan
    complete_soil_data = complete_soil_data.dropna()
    # Ensure 'muid' column is 1-dimensional and consistent
    complete_soil_data['muid'] = complete_soil_data['muid'].astype(int).reset_index(drop=True)
    assert "muid" in complete_soil_data.columns, "muid column not found in complete_soil_data"
    complete_soil_data = complete_soil_data.dropna()
    # Group by 'muid' and calculate the mean for numeric columns
    
    grouped = complete_soil_data.groupby('muid').mean()
    print(f"Range of soil_k values: {complete_soil_data[soil_property].min()} - {complete_soil_data[soil_property].max()}")
    # Load gssurgo grid

    gssurgo_grid, unique_muid, no_value = load_gssurgo_grid(gssurgo_raster_path)
    create_h5py_file(database_path)
    # Loop over each property and plot
    column = soil_property
    # Create a copy of the gssurgo_grid for each column

    _gssurgo_grid = replace_nan_with_no_value(no_value, gssurgo_grid, unique_muid, column, grouped)

    # Plot the updated gssurgo_grid for the current column
    plot_gssurgo_grid(_gssurgo_grid, column)


    ### write into an existing hdf5 file
    add_to_hdf5(column, _gssurgo_grid, database_path)
            

if __name__ == '__main__':

    swat_soil_path = "database/swatplus_soils.sqlite"
    
    generate_ssurgo_csv(swat_soil_path)
    
    path = "database/ssurgo.csv"
    
    soil_data = ['soil_k', 'dp', 'bd', 'awc', 'carbon',
       'clay', 'silt', 'sand', 'rock', 'alb', 'usle_k', 'ec', 'caco3', 'ph',
       'dp_tot']
    
    RESOLUTION = 30
    
    gssurgo_raster_path = f'/data/MyDataBase/SWATGenXAppData/all_rasters/gSURRGO_swat_{RESOLUTION}m.tif'
    database_path=f'/data/MyDataBase/HydroGeoDataset_ML_{RESOLUTION}.h5'
    
    _complete_soil_data = pd.read_csv(path)
    
    for soil_property in soil_data:
        # Load data
        process_soil_data(_complete_soil_data, soil_property, gssurgo_raster_path, database_path)
        