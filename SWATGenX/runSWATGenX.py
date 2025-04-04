from SWATGenX.SWATGenXCommand import SWATGenXCommand
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
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


    LEVEL = "huc12"

    MODEL_NAME = "SWAT_MODEL_Web_Application"

    if LEVEL == "huc12":

        print("Reading the station names from the camel_hydro.txt")
        #time.sleep(5)
        #station_names = gauge_id = pd.read_csv(SWATGenXPaths.camel_hydro_path, sep=";", dtype={"gauge_id": str})['gauge_id'].values

    elif LEVEL == "huc8":
        
        huc8_list = [
            '04050001', '04050002', '04050003', '04050004', '04050005', '04050006', '04050007', 
            '04060101', '04060102', '04060103', '04060104', '04060105', '04070003', '04070004', 
            '04070005', '04070006', '04070007', '04080101', '04080102', '04080103', '04080104', 
            '04080201', '04080202', '04080203', '04080204', '04080205', '04080206', '04090001', 
            '04090003', '04090004', '04100001', '04100002', '04100013'
        ]

    else:
        selected_list = ["HUC4"]

    swatgenx_config = {
        "overwrite": True,
        "database_dir": "/data/SWATGenXApp/GenXAppData/",
        "LEVEL": LEVEL,
        "landuse_product": "NLCD",
        "landuse_epoch": "2021",
        "ls_resolution": "250",
        "dem_resolution": "30",
        "station_name": ['04127200'],
        "MODEL_NAME": MODEL_NAME,
        "MAX_AREA": 1500,
        "MIN_AREA": 500,
        "GAP_percent": 10,
        "username": "admin",
    }

    swat_commander = SWATGenXCommand(swatgenx_config)
    swat_commander.execute()