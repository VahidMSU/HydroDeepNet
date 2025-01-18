import pandas as pd
import os
import logging
try:
    from SWATGenX.read_VPUID import find_VPUID
except:
    from read_VPUID import find_VPUID
import geopandas as gpd
from shapely.geometry import Point
# Set up loggings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def find_station_region(search_term, governmental_boundries_path, USGS_CONUS_shp):

    US_BOUNDARIES = gpd.read_file(governmental_boundries_path, layer="GU_CountyOrEquivalent")
    USGS_CONUS = gpd.read_file(USGS_CONUS_shp)
    available_sites_path = "/data/SWATGenXApp/GenXAppData/USGS/all_VPUIDs.csv"
    available_sites = pd.read_csv(available_sites_path, dtype={"site_no": str})
    USGS_CONUS = USGS_CONUS[USGS_CONUS.site_no.isin(available_sites.site_no)]
    
    countries = []
    states = []
    site_numbers = []
    site_names = []
    VPUIDs = []
    coordinates = []

    counter = 1
    # Loop through unique site names in usgs_stations
    for name in USGS_CONUS.station_nm.unique():
        if search_term in name.lower():
            logging.info("=========================================")
            logging.info(f"{counter}-Site name: {name}")
            site_number = USGS_CONUS[USGS_CONUS.station_nm == name].site_no.values[0]
            logging.info(f"{counter}-Site number: {site_number}")
            # Find VPUID
            VPUID = find_VPUID(site_number)
            logging.info(f"{counter}-VPUID: {VPUID}")
            # Get coordinates
            x = USGS_CONUS[USGS_CONUS.site_no == site_number].geometry.x.values[0]
            y = USGS_CONUS[USGS_CONUS.site_no == site_number].geometry.y.values[0]
            logging.info(f"{counter}-Coordinates: {x:.2f}, {y:.2f}")
            ### find COUNTY_NAME in US_BOUNDARIES by using x and y
            COUNTY = US_BOUNDARIES[US_BOUNDARIES.geometry.contains(Point(x, y))].COUNTY_NAME.values[0]
            STATE = US_BOUNDARIES[US_BOUNDARIES.geometry.contains(Point(x, y))].STATE_NAME.values[0]
            logging.info(f"{counter}-State: {STATE}")
            logging.info(f"{counter}-County: {COUNTY}")
            logging.info("=========================================")
            counter = counter + 1
            countries.append(COUNTY)
            states.append(STATE)
            site_numbers.append(site_number)
            site_names.append(name)
            VPUIDs.append(VPUID)
            coordinates.append(f"{x:.2f}, {y:.2f}")




    return pd.DataFrame(
        {
            "SiteNumber": site_numbers,
            "SiteName": site_names,
            "VPUID": VPUIDs,
            "State": states,
            "County": countries,
            "Coordinates": coordinates,
        }
    )

if __name__ == "__main__":
    governmental_boundries_path =  "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"
    USGS_stations_path = "/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv"
    USGS_CONUS_shp = "/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.shp"
    search_term = "metal"
    df = find_station_region(search_term, governmental_boundries_path, USGS_stations_path, USGS_CONUS_shp)