import os
import shutil
import logging
import sys
from shapely.geometry import mapping

sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT')
sys.path.append(r'/data/MyDataBase/SWATGenXAppData/codes/ModelProcessing')
import geopandas as gpd
from NHDPlus_SWAT.SWATGenXCommand import SWATGenXCommand
from NHDPlus_SWAT.integrate_streamflow_data import integrate_streamflow_data
from ModelProcessing.core import process_SCV_SWATGenXModel

def single_model_creation(site_no, ls_resolution, dem_resolution, calibration_flag, validation_flag, sensitivity_flag, cal_pool_size, sen_pool_size, sen_total_evaluations, num_levels, max_cal_iterations, verification_samples):
    logging.info(f"Starting model creation for site_no: {site_no}")
    BASE_PATH = os.getenv('BASE_PATH', '/data/MyDataBase/SWATGenXAppData/')
    LEVEL = "huc12"
    MAX_AREA = 5000
    MIN_AREA = 10
    GAP_percent = 10

    landuse_product = "NLCD"
    landuse_epoch = "2021"

    station_name = site_no
    functionality = True
    cal_functionality = True

    #try:
    if functionality:
        model_path = SWATGenXCommand(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent, landuse_product, landuse_epoch, ls_resolution, dem_resolution, station_name, MODEL_NAME='SWAT_MODEL_Web_Application', single_model=True,multiple_model_creation=False, target_VPUID=None)

    if cal_functionality:
        process_SCV_SWATGenXModel(NAME=station_name, sensitivity_flag=sensitivity_flag, calibration_flag=calibration_flag, verification_flag=validation_flag, START_YEAR=2015, END_YEAR=2022, nyskip=3, sen_total_evaluations=sen_total_evaluations, sen_pool_size=sen_pool_size, num_levels=num_levels, cal_pool_size=cal_pool_size, max_cal_iterations=max_cal_iterations, termination_tolerance=10, epsilon=0.0001, Ver_START_YEAR=2004, Ver_END_YEAR=2022, Ver_nyskip=3, range_reduction_flag=False, pet=2, cn=1, no_value=1e6, verification_samples=25)

    output_path = os.path.join("D:/Generated_models", f"{site_no}")
    os.makedirs("D:/Generated_models", exist_ok=True)
    shutil.make_archive(output_path, 'zip', model_path)
    logging.info(f"Model creation successful for site_no: {site_no}")
    return f"{output_path}.zip"

   # except Exception as e:
   #     logging.error(f"Error during model creation for site_no {site_no}: {e}")
   #     return "Failed to create SWAT+ model"


def get_huc12_geometries(list_of_huc12s):
    #logging.info(f"Getting geometries for HUC12s: {list_of_huc12s}")
    ### list of huc12 are like:  ['020200030604', '020200030603', '020200030601', '020200030602', '020200030605']
    VPUID = list_of_huc12s[0][:4]
    print(VPUID)
    # Read the shapefile
    path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
    GBD_path = os.listdir(path)
    GBD_path = [x for x in GBD_path if x.endswith('.gdb')][0]
    GBD_path = os.path.join(path, GBD_path)
    gdf = gpd.read_file(GBD_path, layer = "WBDHU12")

    gdf.rename(columns = {'HUC12': 'huc12'}, inplace = True)
    gdf = gdf[gdf['huc12'].isin(list_of_huc12s)]
    return gdf['geometry'].apply(mapping).tolist()