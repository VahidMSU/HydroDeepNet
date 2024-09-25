import arcpy
import os
import itertools
import time
import psutil
import pandas as pd
import numpy as np
import logging
from multiprocessing import Pool
import os
from functools import partial
from multiprocessing import Process, Queue, Semaphore
import multiprocessing
import rasterio
from queue import Empty

def read_hru_wb_mon_txt(hru_wb_mon_txt, chunksize=100000):
    # Check if the file exists
    if not os.path.exists(hru_wb_mon_txt):
        raise FileNotFoundError(f"File not found: {hru_wb_mon_txt}")

    # Read the first few lines to capture the column names
    with open(hru_wb_mon_txt, 'r') as f:
        lines = f.readlines()
        columns = lines[1].split()  # Assuming second line contains column names

    # Define appropriate data types for the columns to save memory
    dtype = {
        'gis_id': 'int32',  # Assuming GIS ID is an integer
        'perc': 'float32',  # Assuming percentage is a float
        'yr': 'int16',      # Year should fit in an int16 (e.g., 2000-2099)
        'mon': 'int8'       # Month should fit in an int8 (range 1-12)
    }

    # Process the file in chunks with the specified data types
    chunks = pd.read_csv(hru_wb_mon_txt, 
                         sep='\s+',                # Assuming whitespace delimiter
                         names=columns,            # Use the extracted column names
                         skiprows=3,               # Skip the first two header lines
                         chunksize=chunksize,      # Read in chunks of 100,000 rows
                         dtype=dtype,              # Specify data types
                         low_memory=False)         # Process in chunks to avoid memory issues

    # Combine all chunks into a single DataFrame (if necessary)
    df = pd.concat(iter(chunks))
    print(f"Read {len(df)} rows from {os.path.basename(hru_wb_mon_txt)}")
    
    # Return only the required columns
    return df[['gis_id', 'perc', 'yr', 'mon']]

def generate_reference_raster(NAME):

    hru_shape_path = f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_MODEL/Watershed/Shapes/hrus2.shp"
    reference_raster_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized/hru_30m.tif"

    if os.path.exists(reference_raster_path):
        logging.info(f"Reference raster already exists: {os.path.basename(reference_raster_path)}")
        return reference_raster_path
    if not os.path.exists(hru_shape_path):
        logging.error(f"File not found: {hru_shape_path}")
        return None

    arcpy.env.overwriteOutput = True    
    os.makedirs(f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized", exist_ok=True)
    arcpy.env.workspace = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/hru_rasterized"
    arcpy.PolygonToRaster_conversion(hru_shape_path, "HRUS", "hru_30m.tif", cellsize=30)

    return reference_raster_path

def convert_numpy_array_to_raster(src_array_revised, reference_raster_path, output_path):
    # Set the ArcGIS environment settings
    arcpy.env.workspace = os.path.dirname(output_path)
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster = reference_raster_path
    arcpy.env.cellSize = reference_raster_path
    arcpy.env.extent = reference_raster_path
    arcpy.env.outputCoordinateSystem = reference_raster_path
    reference_raster = arcpy.Raster(reference_raster_path)
    lower_left = reference_raster.extent.lowerLeft
    cell_size = reference_raster.meanCellWidth  # Make sure this comes from the reference raster
    print(f"Numpy to raster: {os.path.basename(output_path)}")
    out_raster = arcpy.NumPyArrayToRaster(src_array_revised, lower_left, cell_size, cell_size)  # Explicitly set cell size
    # Save the raster to the output path
    
    out_raster.save(output_path)

    return output_path



def merge_with_reference_raster(df_filtered, src_array, no_value, yr, mon):
    print(f"merging data {yr} {mon}")

    # Ensure 'gis_id' and 'perc' columns exist
    assert "gis_id" in df_filtered.columns, "gis_id column not found in the DataFrame"
    assert "perc" in df_filtered.columns, "perc column not found in the DataFrame"

    # Create a dictionary to map gis_id to perc
    gis_id_to_perc = dict(zip(df_filtered['gis_id'], df_filtered['perc']))

    # Create a copy of the source array to modify
    result_array = np.copy(src_array)

    # Create a boolean mask where gis_id in src_array matches gis_ids from df_filtered
    mask = np.isin(result_array, df_filtered['gis_id'])

    # Apply the dictionary lookup using NumPy's where and the vectorized `get` method
    unique_gis_ids = df_filtered['gis_id'].unique()
    
    # Only proceed if there are any matching `gis_id`s
    if unique_gis_ids.size > 0:
        # Apply the mapping for each unique `gis_id`
        for gis_id in unique_gis_ids:
            result_array[result_array == gis_id] = gis_id_to_perc.get(gis_id, no_value)

    # Replace `no_value` in result_array with np.nan
    result_array = np.where(result_array == no_value, np.nan, result_array)

    return result_array


def remove_original_file(output_path):
    # Remove the original file after copying
    if arcpy.Exists(output_path):
        try:
            arcpy.Delete_management(output_path)
            print("Deleted original file")
        except Exception as e:
            print(f"Coudn't delete the original file: {output_path}")

def resample_to_250m(output30m, yr):
    print(f"Resampling to 250m: {os.path.basename(output30m)}")
    output250 = output30m.replace("30m", "250m")
    ## reference raster for 250 resolution
    refernce_250m = f"{os.path.dirname(output250)}/recharge_{yr}.tif"
    crs = arcpy.Describe(refernce_250m).spatialReference
    arcpy.env.workspace = os.path.dirname(refernce_250m)
    arcpy.env.snapRaster = refernce_250m
    arcpy.env.cellSize = refernce_250m
    arcpy.env.extent = refernce_250m
    arcpy.env.outputCoordinateSystem = crs
    arcpy.Resample_management(output30m, output250, "250 250", "BILINEAR")


def process_file(mon, yr, NAME, ver, hru_wb_mon_txt, reference_raster_path, src_array, no_value):
    try:
        df = read_hru_wb_mon_txt(hru_wb_mon_txt)
        df_filtered = df[(df['yr'] == yr) & (df['mon'] == mon)]
        print(f"Processing {yr} {mon} for {NAME} and {ver}")
        src_array_revised = merge_with_reference_raster(df_filtered, src_array, no_value, yr, mon)
        output_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/recharg_output_SWAT_gwflow_MODEL/verification_stage_{ver}/recharge_{yr}_{mon}_30m.tif"   
        convert_numpy_array_to_raster(src_array_revised, reference_raster_path, output_path)
        resample_to_250m(output_path, yr)
#        remove_original_file(output_path)
        return True
    except Exception as e:
        print(f"Error processing {yr} {mon} for {NAME} and {ver}: {e}")
        return False

def process_task(task):
    """Helper function to process individual tasks."""
    task()

def generate_tasks(NAME, ver):
    """Generates the tasks for a specific NAME and version."""
    hru_wb_mon_txt = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_{ver}/hru_wb_mon.txt"

    if not os.path.exists(hru_wb_mon_txt):
        print(f"File not found: {hru_wb_mon_txt}")
        return []

    print(f"Processing {NAME} and version {ver}")
    reference_raster_path = generate_reference_raster(NAME)

    if not os.path.exists(reference_raster_path):
        print(f"Reference raster not found: {reference_raster_path}")
        return []

    with rasterio.open(reference_raster_path) as src:
        src_array = src.read(1)
        no_value = src.nodata

    return [
        partial(
            process_file,
            mon,
            yr,
            NAME,
            ver,
            hru_wb_mon_txt,
            reference_raster_path,
            src_array,
            no_value,
        )
        for yr, mon in itertools.product(range(2000, 2020), range(1, 13))
    ]

def worker(queue, semaphore):
    """Worker function to process tasks from the queue."""
    while True:
        try:
            task = queue.get(timeout=5)  # Timeout to avoid infinite block
            if task is None:
                break
            semaphore.acquire()  # Acquire semaphore before processing
            process_task(task)   # Process the task
            semaphore.release()  # Release semaphore after processing
        except Empty:
            break

if __name__ == "__main__":
    print("Starting the process")
    
    # Define the number of processes and semaphore limit
    num_workers = 60
    semaphore_limit = 60

    NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
    NAMES.remove("log.txt")
    NAMES.reverse()

    # Initialize the Queue and Semaphore
    task_queue = Queue()
    semaphore = Semaphore(semaphore_limit)

    # Create worker processes
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue, semaphore))
        p.start()
        processes.append(p)

    # Iterate over the second half of NAMES and create tasks
    for NAME in NAMES:
        for ver in range(1,4):
            tasks = generate_tasks(NAME, ver)
            for task in tasks:
                task_queue.put(task)

    # Signal the workers to stop after all tasks are processed
    for _ in range(num_workers):
        task_queue.put(None)

    # Join the worker processes
    for p in processes:
        p.join()

    print("Finished processing all tasks.")
