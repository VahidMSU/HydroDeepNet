from NHDPlus_SWAT.SWATGenXCommand import SWATGenXCommand
import time
import pandas as pd
import os
import logging

"""
/***************************************************************************
    SWATGenX
                            -------------------
        begin                : 2023-05-15
        copyright            : (C) 2024 by Vahid Rafiei
        email                : rafieiva@msu.edu
***************************************************************************/

/***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************/
"""

def check_station(station_name):

    FPS_sites_path = "/data/MyDataBase/SWATGenXAppData/USGS/FPS_States_and_Territories.csv"
    FPS_sites = pd.read_csv(FPS_sites_path, skiprows=1)
    FPS_status = FPS_sites[FPS_sites.SiteNumber == station_name]

    if len(FPS_status) != 0:

        logging.info(f"Site number {station_name} is valid")
        logging.info(f"SiteName: {FPS_status.SiteName.values[0]}")
        logging.info(f"Status: {FPS_status.Status.values[0]}")
        logging.info(f"USGSFunding: {FPS_status.USGSFunding.values[0]}")
    else:
        logging.error(f"Site number {station_name} is not among FPS lists")

def single_model_creation(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent,
                        landuse_product, landuse_epoch, ls_resolution, dem_resolution,
                        station_name, user_input):
    BASE_PATH = r'/data/MyDataBase/SWATGenXAppData/'
    LEVEL = "huc12"
    MAX_AREA = 1000
    MIN_AREA = 100
    GAP_percent = 10
    MODEL_NAME = "SWAT_MODEL"

    landuse_product = "NLCD"
    landuse_epoch = "2021"
    ls_resolution = "100"
    dem_resolution = "30"

    station_name = user_input
    SWATGenXCommand(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent,
                    landuse_product, landuse_epoch, ls_resolution, dem_resolution,
                    station_name,MODEL_NAME,
                    single_model=True, random_model_selection=False, multiple_model_creation=False,
                    target_VPUID=None)

if __name__ == "__main__":
    logfile_path = "/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/SWATGenX.log"
    with open(logfile_path, 'w') as file:
        file.write(" Processing SWATGenX \n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=logfile_path)
    BASE_PATH = r'/data/MyDataBase/SWATGenXAppData/'
    LEVEL = "huc12"
    MAX_AREA = 3500
    MIN_AREA = 50
    GAP_percent = 10
    landuse_product = "NLCD"
    landuse_epoch = "2021"
    ls_resolution = "30"
    dem_resolution = "30"
    MODEL_NAME = "SWAT_MODEL_gssurgo"
    station_names = ['04166000', '04166300', '04161540' ,'04167000', '04166100' ,'04161000']
    station_name = '04128990'
    single_model = True
    #for station_name in station_names:
    check_station(station_name)
    multiple_model_creation = False    ### parallel processing when extracting NSRDB does not work very well. so here although the parallel processing is enabled, we send one station at a time
    target_VPUIDs = os.listdir("/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/")

    for target_VPUID in target_VPUIDs:
        if target_VPUID not in ["0405", "0406", "0407", "0408","0409", "0410"]:
            logging.info(f"Processing VPUID: {target_VPUID}")

            SWATGenXCommand(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent,
                            landuse_product, landuse_epoch, ls_resolution, dem_resolution,
                            station_name, MODEL_NAME,
                            single_model, multiple_model_creation,
                            target_VPUID)
            if single_model:
                break
