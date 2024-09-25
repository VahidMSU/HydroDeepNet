import arcpy
from arcpy import env
from arcpy.sa import *
from arcpy.ga import *
from arcpy.management import Delete
import geopandas as gpd
import pandas as pd
import os
import time
import numpy as np
from sklearn.neighbors import KDTree
import gc
from shapely.geometry import box
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copy2
import time
import arcpy

def process_parameter(parameter):
    arcpy.env.overwriteOutput=True
    cell_size = 250

    print(f"Processing {parameter} for {cell_size}m resolution")


    crs_setting='EPSG:26990'

    BASE_PATH = r"/data/MyDataBase/SWATGenXAppData/Well_data_krigging/"


    # Path to your point shapefile
    groundwater_data_path = (
        f"{BASE_PATH}WaterWells_Michigan_combined_projected_gr.pk1"
    )
  #  groundwater_data_path='/data/MyDataBase/SWATGenXAppData/Grid/grid_points_well_obs_with_geometry.pk1'

    # Measure read time
    read_start_time = time.time()
    GWD = gpd.GeoDataFrame(pd.read_pickle(groundwater_data_path), crs=crs_setting, geometry='geometry')
    read_end_time = time.time()
    read_elapsed_time = read_end_time - read_start_time
    print(f"Read Time: {read_elapsed_time} seconds")

    # Some operations
    GWD_cleaned = GWD[~GWD[parameter].isna()][[parameter, 'geometry']].reset_index(drop=True)
    GWD_cleaned = GWD_cleaned.drop_duplicates(subset='geometry').reset_index(drop=True)

    # Measure write time
    workspace= r"R:/Krigging_work_space"
    os.makedirs(workspace, exist_ok=True)

    groundwater_shape_path =os.path.join(workspace, parameter)    ### writing in RAM instream of on DISK

    write_start_time = time.time()
    GWD_cleaned.to_file(groundwater_shape_path)
    write_end_time = time.time()
    write_elapsed_time = write_end_time - write_start_time
    print(f"Write Time: {write_elapsed_time} seconds")

    env.workspace = workspace
    env.overwriteOutput = True
    in_features = os.path.join(groundwater_shape_path , f"{parameter}.shp")

    z_field = parameter

    # Initialize variablesfor krigging
    print('parameter selection')

    transformation_type = "EMPIRICAL"
    max_local_points = 250
    overlap_factor = 2
    number_semivariograms = 150
    semivariogram_model_type = "EXPONENTIAL"

    out_ga_layer = ""

    out_raster_error = f"kriging_stderr_{parameter}_{cell_size}m.tif"
    out_raster_pred = f"kriging_output_{parameter}_{cell_size}m.tif" 


    TARGET_PATH_TO_COPY = r"D:\MyDataBase\Well_data_krigging\Krigging_work_space"


    if arcpy.Exists(os.path.join(TARGET_PATH_TO_COPY, out_raster_pred)):
        print(f"the following file exists:{out_raster_pred}")
    else:
        print('start krigging')
        read_start_time = time.time()
        arcpy.EmpiricalBayesianKriging_ga(in_features, z_field, out_ga_layer, out_raster_pred, cell_size, 
                                          transformation_type, max_local_points, overlap_factor, 
                                          number_semivariograms, "", 'PREDICTION', "", "", "", 
                                          semivariogram_model_type)
        read_end_time = time.time()
        read_elapsed_time = read_end_time - read_start_time
        print(f'Krigging time:{read_elapsed_time}')
        arcpy.Copy_management(os.path.join(workspace, out_raster_pred), os.path.join(TARGET_PATH_TO_COPY, os.path.basename(out_raster_pred)))



    if arcpy.Exists(os.path.join(TARGET_PATH_TO_COPY, out_raster_error)):
        print(f"the following file exists:{out_raster_pred}")
    else:
        print('start calculating errors...')
        read_start_time = time.time()
        arcpy.EmpiricalBayesianKriging_ga(in_features, z_field, out_ga_layer, out_raster_error, cell_size, 
                          transformation_type, max_local_points, overlap_factor, 
                          number_semivariograms, "", "PREDICTION_STANDARD_ERROR", "", "", "", 
                          semivariogram_model_type)
        read_end_time = time.time()
        read_elapsed_time = read_end_time - read_start_time
        print(f'Error Krigging time:{read_elapsed_time}')
        arcpy.Copy_management(os.path.join(workspace, out_raster_error), os.path.join(TARGET_PATH_TO_COPY, os.path.basename(out_raster_error)))

            
def run_script(parameter):
    process_parameter(parameter)