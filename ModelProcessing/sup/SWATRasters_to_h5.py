import h5py
import numpy as np
import os
import pandas as pd
import rasterio
import time
from multiprocessing import Process
from functools import partial
from multiprocessing import Lock



def read_raster(path):
    with rasterio.open(path, 'r') as src:
        data = src.read(1)
        novalues = src.nodata
        data = np.where(data == novalues, -999, data)
        return data


def write_to_h5(SWATplus_output, data, group_name, name, ver, NAME):
        ## lock h5
    h5_lock = Lock()    
    with h5_lock:
        with h5py.File(SWATplus_output, "a") as f:
            if f'hru_wb_30m/2000/1/perc' not in f:
                print(f"Mask not found in {SWATplus_output}")
                return
            mask = f[f'hru_wb_30m/2000/1/perc'][:]
            print(f"Mask shape: {mask.shape}")
            ## make sure the mask is the same shape as the data
            max_rows = min(mask.shape[0], data.shape[0])
            max_cols = min(mask.shape[1], data.shape[1])
            data = data[:max_rows, :max_cols]
            mask = mask[:max_rows, :max_cols]

            data = np.where(mask == -999, -999, data)
            if f'{group_name}/{name}' in f:
                del f[f'{group_name}/{name}']
            f.create_dataset(f"{group_name}/{name}", data=data, compression="gzip", compression_opts=9)
            ### assert the data is written correctly
            assert np.allclose(data, f[f'{group_name}/{name}'][:]), f"Data not written correctly to {SWATplus_output}"

            print(f"Data written to {SWATplus_output} under {name}")
            print(f"Range of values: {np.nanmin(data)} to {np.nanmax(data)}")
            print(f"Shape of data: {data.shape}")
            print(f"Number of NaN values: {np.sum(np.isnan(data))}")
            print(f"Number of no data values: {np.sum(data == -999)}")

            ### add author, date, resolution and description
                
            f.attrs['author'] = "Vahid Rafiei"
            f.attrs['date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.attrs['resolution'] = "30m"
            f.attrs['description'] = f"SWAT+ outputs for USGS Streamflow station {NAME}, verification stage {ver}"


def process_and_write_raster(swat_base_path, SWATplus_output, ver, raster_name, file_name, NAME):
    path = f"{swat_base_path}{raster_name}/{file_name}"
    assert os.path.exists(path), f"File not found: {path}"
    data = read_raster(path)
    print(f"shape of {raster_name}: {data.shape}")
    write_to_h5(SWATplus_output, data, raster_name, file_name.replace(".tif", ""), ver, NAME)


def add_gssurgo_dem_landuse(NAME, ver = 0):


    """"
    
    This script add soil, landuse, dem, and other rasters to the SWATplus_output.h5 file

    """

    swat_base_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Rasters/"
    SWATplus_output = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"

    raster_files = [
        ("Soil", "soil_30m.tif"),
        ("Landuse", "landuse_30m.tif"),
        ("DEM", "dem.tif"),
        ("DEM", "demslp.tif"),
        ("Soil", "clay_30m.tif"),
        ("Soil", "silt_30m.tif"),
        ("Soil", "awc_30m.tif"),
        ("Soil", "bd_30m.tif"),
        ("Soil", "soil_k_30m.tif"),
        ("Soil", "dp_30m.tif"),
        ("Soil", "carbon_30m.tif"),
        ("Soil", "rock_30m.tif"),
        ("Soil", "ph_30m.tif"),
        ("Soil", "caco3_30m.tif"),
        ("Soil", "ec_30m.tif"),
        ("Soil", "alb_30m.tif"),
    ]
    processes = []
    for raster_name, file_name in raster_files:
        ### verify the file exists
        path = f"{swat_base_path}{raster_name}/{file_name}"
        print(f"Processing {path}")
        assert os.path.exists(path), f"File not found: {file_name}"
        process_and_write_raster(swat_base_path, SWATplus_output, ver, raster_name, file_name, NAME)
     

    print(f"Finished processing {NAME} verification stage {ver}")
print("Finished processing all data")


if __name__ == "__main__":
    swat_base_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    NAMES = os.listdir(swat_base_path)
    NAMES.remove("log.txt")
    for NAME in NAMES:
        for ver in range(0, 6):
            add_gssurgo_dem_landuse(NAME, ver)
            print(f"Finished processing {NAME} verification stage {ver}")
