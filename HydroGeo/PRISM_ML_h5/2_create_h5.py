import rasterio
import h5py
import numpy as np
import os
from rasterio.warp import transform
import logging
from filelock import FileLock, Timeout
import time
from multiprocessing import Pool
from functools import partial
logging.basicConfig(level=logging.INFO)
# Create the HDF5 file structure
def create_PRISM_h5(PRISM_h5_path):
    variables = ["ppt", "tmax", "tmin"]
    years = range(1990, 2023)
    with h5py.File(PRISM_h5_path, "w") as h5_file:
        for var in variables:
            group = h5_file.create_group(var)
            for year in years:
                group.create_group(str(year))

# Create metadata for the HDF5 file
def create_metadata(h5_path, reference_raster):
    with h5py.File(h5_path, "a") as h5_file:
        metadata = h5_file.create_group("metadata")
        with rasterio.open(reference_raster) as src:
            metadata.create_dataset("data_type", data="float32")
            metadata.create_dataset("no_data_value", data=[-999], dtype='int32')
            metadata.create_dataset("projection", data=str(src.crs))
            metadata.create_dataset("cell_size", data=src.res[0])
            metadata.create_dataset("author", data="Vahid Rafiei")
            metadata.create_dataset("author_email", data="rafieiva@msu.edu")
            metadata.create_dataset("date_created", data="2024-07-29")
            metadata.create_dataset("source", data="PRISM")


            # Calculate latitude and longitude arrays in EPSG:4326
            transform_affine = src.transform
            height, width = src.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(transform_affine, rows, cols)
            xs = np.array(xs)
            ys = np.array(ys)

            lons, lats = transform(src.crs, 'EPSG:4326', xs.flatten(), ys.flatten())
            lons = np.array(lons).reshape(height, width)
            lats = np.array(lats).reshape(height, width)

            metadata.create_dataset("latitude", data=lats)
            metadata.create_dataset("longitude", data=lons)

def read_stack_prism_h5(var, year, PRISM_h5_path):
    path = f"E:/PRISM/{var}/{year}"
    if not os.path.exists(path):
        logging.error(f"Path does not exist: {path}")
        return
    tifs = os.listdir(path)

    if len(tifs) <1460:
        logging.error(f"Not all rasters are available for {var} {year}")
        return

    tifs = [os.path.join(path, tif) for tif in tifs if tif.endswith(".tif")]

    ## read the raster as numpy
    all_arrays = []
    for i, tif in enumerate(tifs):
        logging.info(f"Reading {i+1} of {len(tifs)}")
        with rasterio.open(tif) as src:
            data = src.read(1)
            ### replace no data value with -999
            data[data==src.nodata] = -999
            all_arrays.append(data)

    while os.path.exists(path+".lock"):
        logging.info(f"Waiting for the lock to be released for {PRISM_h5_path}")
        ## choice a random betwen 20 to 60
        time.sleep(random.randint(20, 60))
    return add_array_to_h5(PRISM_h5_path, var, year, np.stack(all_arrays, axis=0))

def add_array_to_h5(h5_path, var, year, array):
    lock_path = f"{h5_path}.lock"  # Create a lock file path
    with FileLock(lock_path):
        with h5py.File(h5_path, "a") as h5_file:
            ### with uint16, the compression is about 1/3
            h5_file[f"{var}/{year}"].create_dataset("data", data=array, compression="gzip", compression_opts=9, dtype="float32")

def process_var_year(args):
    var, year, PRISM_h5_path = args
    read_stack_prism_h5(var, year, PRISM_h5_path)

if __name__ == "__main__":
    PRISM_h5_path = "Z:/MyDataBase/PRISM_ML_250m.h5"
    reference_raster = "/data/MyDataBase/SWATGenXAppData/all_rasters/DEM_250m.tif"
    if os.path.exists(PRISM_h5_path):
        os.remove(PRISM_h5_path)
    create_PRISM_h5(PRISM_h5_path)
    create_metadata(PRISM_h5_path, reference_raster)

    remaining_count = 0  # Initialize a counter for remaining datasets

    while True:
        all_filled = True  # Flag to check if all data is filled
        remaining_count = 0  # Reset the counter at the start of each loop

        tasks = []
        for var in ["ppt", "tmax", "tmin"]:
            for year in range(1990, 2023):
                logging.info(f"Processing {var} {year}")

                with h5py.File(PRISM_h5_path, "a") as h5_file:
                    # Check if the data exists and is not empty
                    d = h5_file[f"{var}/{year}"].get("data")
                    if d is not None and d.size > 0:
                        logging.info(f"{var} {year} already exists")
                        continue
                    else:
                        all_filled = False  # Set flag to False if data needs to be filled
                        remaining_count += 1  # Increment the counter for unfilled datasets
                        tasks.append((var, year,PRISM_h5_path))

        # Process tasks in parallel
        with Pool(50) as p:

            p.map(process_var_year, tasks)
            time.sleep(2)

        # Write the remaining count to a text file
        with open("remaining_count.txt", "w") as f:
            f.write(f"Number of unfilled datasets: {remaining_count}")

        if all_filled:
            break  # Terminate loop if all data is filled

    logging.info("All data has been filled.")