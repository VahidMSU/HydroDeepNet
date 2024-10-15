import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import datetime
import os
from functools import partial
import rasterio
from multiprocessing import Process
import glob

def get_all_VPUIDs():
    path = "/data/SWATGenXApp/GenXAppData/NHDPlusData/NHDPlus_VPU_National/"
    ## get all file names ending with .zip
    files = glob.glob(f"{path}*.zip")

    VPUIDs = [os.path.basename(file).split('_')[2] for file in files]

    print(VPUIDs)
    return VPUIDs

# Define the paths
PRISM_data_path = "/data/SWATGenXApp/GenXAppData/PRISM"
NHDPlusData_path = "/data/SWATGenXApp/GenXAppData/NHDPlusData"
PRISM_grid_path = "/data/SWATGenXApp/GenXAppData/PRISM/prism_4km_mesh/prism_4km_mesh.shp"
PRISM_dem_path = "/data/SWATGenXApp/GenXAppData/PRISM/PRISM_us_dem_4km_bil/PRISM_us_dem_4km_bil.bil"

def clip_PRISM_by_VPUID(VPUID):
    """This function clips the PRISM data by the extent of the WBDHU8 of the VPUID"""
    output_path = f"{PRISM_data_path}/VPUID/{VPUID}/PRISM_grid.shp"
    SWAT_MODEL_PRISM_path = f'/data/SWATGenXApp/GenXAppData/PRISM/VPUID/{VPUID}/'
    os.makedirs(SWAT_MODEL_PRISM_path, exist_ok=True)
    if not os.path.exists(output_path):
        _extracted_from_clip_PRISM_by_VPUID_7(VPUID, output_path)
    else:
        print(f"Clipped PRISM data for {VPUID} exists.")

    return output_path


# TODO Rename this here and in `clip_PRISM_by_VPUID`
def _extracted_from_clip_PRISM_by_VPUID_7(VPUID, output_path):
    print(f"Clipping PRISM data for {VPUID}...")
    WBDHU8 = pd.read_pickle(fr"{NHDPlusData_path}/SWATPlus_NHDPlus/{VPUID}/WBDHU8.pkl").to_crs("EPSG:4326")
    extent = WBDHU8.total_bounds
    PRISM = gpd.read_file(PRISM_grid_path).to_crs("EPSG:4326")
    clipped_PRISM_HUC12 = PRISM.cx[extent[0]:extent[2], extent[1]:extent[3]]
    os.makedirs(f"{PRISM_data_path}/VPUID/{VPUID}", exist_ok=True)
    WBDHU12 = pd.read_pickle(fr"{NHDPlusData_path}/SWATPlus_NHDPlus/{VPUID}/WBDHU12.pkl").to_crs("EPSG:4326")
    PRISM_grid_hu12 = gpd.overlay(clipped_PRISM_HUC12, WBDHU12[['huc12','geometry']], how='intersection')
    PRISM_grid_hu12.to_file(output_path)
    print(f"Clipping PRISM data for {VPUID} is done.")

def generating_swatplus_pcp(VPUID, PRISM_VPUID_grid, datasets, years,SWAT_MODEL_PRISM_path ):

    """Generate SWAT+ precipitation data for a given VPUID"""
    # Get elevation once
    with rasterio.open(PRISM_dem_path) as src:
        elev_data = src.read(1)

    for row, col in zip(PRISM_VPUID_grid.row, PRISM_VPUID_grid.col):
        array = []
        #print(year)
        for year in years:
            time_series = datasets[year]['data'][:, row, col]
            array = np.append(array, time_series)

        #print(f"Processing {row} and {col} is done.")
        #print(array.shape)

        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        date_df = pd.DataFrame(date_range, columns=['date'])

        date_df['YEAR'] = date_df['date'].dt.year
        date_df['DAY'] = date_df['date'].dt.dayofyear
        date_df["ppt"] = array

        #print(date_df.head())

        title = f"PRISM 4sqkm resolution grid for VPUID: {VPUID}, row: {row}, col: {col}"
        headers = "nbyr\ttstep\tlat\tlon\telev"

        nbyr = 2022 - 1990 + 1
        tstep = 0
        lat = PRISM_VPUID_grid[(PRISM_VPUID_grid.row==row) & (PRISM_VPUID_grid.col==col)].lat.values[0]
        lon = PRISM_VPUID_grid[(PRISM_VPUID_grid.row==row) & (PRISM_VPUID_grid.col==col)].lon.values[0]
        elev = elev_data[row, col]

        if SWAT_MODEL_PRISM_path is None:
            path_to_save = os.path.join(f"{PRISM_data_path}/VPUID/{VPUID}/", f"r{row}_c{col}.pcp")
        else:
            path_to_save = os.path.join(SWAT_MODEL_PRISM_path, f"r{row}_c{col}.pcp")

        with open(path_to_save, 'w') as file:
            file.write(f"{title}\n")
            file.write(f"{headers}\n")
            file.write(f"{nbyr} {tstep} {lat} {lon} {elev}\n")
            for index, row in date_df.iterrows():
                year = row['YEAR']
                day = f"{row['DAY']:03}"
                ppt = f"{row['ppt']:.2f}"
                file.write(f"{year}\t{day}\t{ppt}\n")

def generating_swatplus_tmp(VPUID, PRISM_VPUID_grid, datasets_max,datasets_min ,years,SWAT_MODEL_PRISM_path):
    """Generate SWAT+ temperature data for a given VPUID"""
    with rasterio.open(PRISM_dem_path) as src:
        elev_data = src.read(1)

    headers = "nbyr\ttstep\tlat\tlon\telev"
    tstep = 0
    for row, col in zip(PRISM_VPUID_grid.row, PRISM_VPUID_grid.col):
        TYPES = ["tmax", "tmin"]

        array_max = []
        array_min = []

        for TYPE in TYPES:
            # print(f"Processing {row} and {col}...")
            time_series = []
            for year in years:
                if TYPE == "tmax":
                    time_series = datasets_max[year]['data'][:, row, col]
                    array_max = np.append(array_max, time_series)
                elif TYPE == "tmin":
                    time_series = datasets_min[year]['data'][:, row, col]
                    array_min = np.append(array_min, time_series)

        start_date = datetime.datetime(years[0], 1, 1)
        end_date = datetime.datetime(years[-1], 12, 31)
        date_range = pd.date_range(start_date, end_date)
        date_df = pd.DataFrame(date_range, columns=['date'])

        date_df['YEAR'] = date_df['date'].dt.year
        date_df['DAY'] = date_df['date'].dt.dayofyear
        date_df['tmax'] = array_max
        date_df['tmin'] = array_min

        # print(date_df.head())
        title = f"PRISM 4sqkm resolution grid for VPUID: {VPUID}, row: {row}, col: {col}"
        nbyr = 2022 - 1990 + 1
        lat = PRISM_VPUID_grid[(PRISM_VPUID_grid.row==row) & (PRISM_VPUID_grid.col==col)].lat.values[0]
        lon = PRISM_VPUID_grid[(PRISM_VPUID_grid.row==row) & (PRISM_VPUID_grid.col==col)].lon.values[0]
        elev = elev_data[row, col]
        path_to_save = f"{PRISM_data_path}/VPUID/{VPUID}/r{row}_c{col}.tmp"

        if SWAT_MODEL_PRISM_path is None:
            path_to_save = os.path.join(f"{PRISM_data_path}/VPUID/{VPUID}/", f"r{row}_c{col}.tmp")
        else:
            path_to_save = os.path.join(SWAT_MODEL_PRISM_path, f"r{row}_c{col}.tmp")

        with open(path_to_save, 'w') as file:
            file.write(f"{title}\n")
            file.write(f"{headers}\n")
            file.write(f"{nbyr} {tstep} {lat} {lon} {elev}\n")
            for index, row in date_df.iterrows():
                year = row['YEAR']
                day = f"{row['DAY']:03}"
                tmax = f"{row['tmax']:.2f}"
                tmin = f"{row['tmin']:.2f}"
                file.write(f"{year}\t{day}\t{tmax}\t{tmin}\n")

def extract_PRISM_parallel(VPUID, list_of_huc12s=None, SWAT_MODEL_PRISM_path=None):
    """This function extracts PRISM data for a given VPUID using parallel processing"""

    PRISM_VPUID_path = clip_PRISM_by_VPUID(VPUID)
    print(f"Extracting PRISM data for {VPUID}...")
    PRISM_VPUID_grid = gpd.read_file(PRISM_VPUID_path)
    #PRISM_VPUID_grid = PRISM_VPUID_grid.drop_duplicates(subset=['row', 'col'])

    if list_of_huc12s is not None:
        os.makedirs(SWAT_MODEL_PRISM_path, exist_ok=True)
        PRISM_VPUID_grid = PRISM_VPUID_grid[PRISM_VPUID_grid['huc12'].isin(list_of_huc12s)]
        _temp_PRISM_grid = gpd.read_file(PRISM_grid_path)
        ### find the geometry of the unqiue rows and cols in the _temp_PRISM_grid based on the row and column from PRISM_VPUID_grid
        _temp_PRISM_grid = _temp_PRISM_grid[_temp_PRISM_grid['row'].isin(PRISM_VPUID_grid['row']) & _temp_PRISM_grid['col'].isin(PRISM_VPUID_grid['col'])]

        _temp_PRISM_grid.to_crs("EPSG:4326").to_file(os.path.join(SWAT_MODEL_PRISM_path,'PRISM_grid.shp'))
    else:
        _temp_PRISM_grid = PRISM_VPUID_grid
    PRISM_VPUID_grid = PRISM_VPUID_grid.drop(columns='geometry')
    ## measure the time for the extraction
    import time
    start = time.time()
    datasets = {}
    YEARS = np.arange(1990, 2023)
    print("Loading PRISM ppt data...")
    for YEAR in YEARS:
        path = f"{PRISM_data_path}/CONUS/ppt/{YEAR}.nc"
        if not os.path.exists(path):
            print(f"{path} does not exist.")
            continue
        print(f"Loading {YEAR}...")
        datasets[YEAR] = xr.open_dataset(path)
    generating_swatplus_pcp(VPUID, _temp_PRISM_grid, datasets, YEARS, SWAT_MODEL_PRISM_path)
    end = time.time()
    print(f"Time taken for the extraction of PRISM ppt data is {end - start} seconds.")

    datasets_max = {}
    datasets_min = {}
    start = time.time()
    for YEAR in YEARS:
        path_max = f"{PRISM_data_path}/CONUS/tmax/{YEAR}.nc"
        path_min = f"{PRISM_data_path}/CONUS/tmin/{YEAR}.nc"
        if not os.path.exists(path_max):
            print(f"{path_max} does not exist.")
            continue
        if not os.path.exists(path_min):
            print(f"{path_min} does not exist.")
            continue
        print(f"Loading {YEAR}...")
        datasets_max[YEAR] = xr.open_dataset(path_max)
        datasets_min[YEAR] = xr.open_dataset(path_min)
    print("PRISM data are loaded.")
    print("Processing PRISM data...")
    generating_swatplus_tmp(VPUID, _temp_PRISM_grid, datasets_max,datasets_min ,YEARS,SWAT_MODEL_PRISM_path)
    end = time.time()

    print(f"Time taken for the extraction of PRISM temperature data is {end - start} seconds.")
if __name__ == "__main__":
    VPUIDs = get_all_VPUIDs()
    processes = []
    extract_PRISM_parallel("0407", list_of_huc12s=None, SWAT_MODEL_PRISM_path="/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0407/huc12/04128990/PRISM")