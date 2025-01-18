from SWATGenX.SWATGenXCommand import SWATGenXCommand
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

    FPS_sites_path = "/data/SWATGenXApp/GenXAppData/USGS/FPS_States_and_Territories.csv"
    FPS_sites = pd.read_csv(FPS_sites_path, skiprows=1)
    FPS_status = FPS_sites[FPS_sites.SiteNumber == station_name]

    if len(FPS_status) != 0:

        logging.info(f"Site number {station_name} is valid")
        logging.info(f"SiteName: {FPS_status.SiteName.values[0]}")
        logging.info(f"Status: {FPS_status.Status.values[0]}")
        logging.info(f"USGSFunding: {FPS_status.USGSFunding.values[0]}")
    else:
        logging.error(f"Site number {station_name} is not among FPS lists")


if __name__ == "__main__":

    logfile_path = "/data/SWATGenXApp/codes/SWATGenX.log"

    with open(logfile_path, 'w') as file:
        file.write(" Processing SWATGenX \n")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=logfile_path)

    LEVEL = "huc12"

    MODEL_NAME = "SWAT_MODEL"

    station_names = ['04166000']#, '04166300', '04161540' ,'04167000', '04166100' ,'04161000']

    huc8_list = [
        '04050001', '04050002', '04050003', '04050004', '04050005', '04050006', '04050007', 
        '04060101', '04060102', '04060103', '04060104', '04060105', '04070003', '04070004', 
        '04070005', '04070006', '04070007', '04080101', '04080102', '04080103', '04080104', 
        '04080201', '04080202', '04080203', '04080204', '04080205', '04080206', '04090001', 
        '04090003', '04090004', '04100001', '04100002', '04100013'
    ]

    single_model = True

    if not single_model:
        selected_list = huc8_list if LEVEL == "huc8" else station_names
    else:
        selected_list = ['04161540'] if LEVEL == "huc8" else ['04166000']

    for station_name in selected_list:
        check_station(station_name)
        swatgenx_config = {
            "database_dir": "/data/SWATGenXApp/GenXAppData/",
            "LEVEL": LEVEL,
            "landuse_product": "NLCD",
            "landuse_epoch": "2021",
            "ls_resolution": "250",
            "dem_resolution": "30",
            "station_name": station_name,
            "MODEL_NAME": MODEL_NAME,
            "single_model": True,
            "MAX_AREA": 85000,
            "MIN_AREA": 50,
            "GAP_percent": 10,
            "region": "12",
           # "target_VPUID": get_all_VPUIDs()
        }

        #SWATGenXCommand(swatgenx_config)
        swat_commander = SWATGenXCommand(swatgenx_config)
        swat_commander.execute()