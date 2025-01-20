import pandas as pd
import os
try:
    from SWATGenX.utils import find_VPUID
    from SWATGenX.SWATGenXLogging import LoggerSetup
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except Exception:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenXLogging import LoggerSetup
    from utils import find_VPUID

import geopandas as gpd
from shapely.geometry import Point

def find_station_region(search_term):
    
    US_BOUNDARIES = gpd.read_file(SWATGenXPaths.governmental_boundries_path, layer="GU_CountyOrEquivalent")
    USGS_CONUS = gpd.read_file(SWATGenXPaths.USGS_CONUS_stations_shape_path)
    available_sites = pd.read_csv(SWATGenXPaths.available_sites_path, dtype={"site_no": str})
    USGS_CONUS = USGS_CONUS[USGS_CONUS.site_no.isin(available_sites.site_no)]
    
    logger = LoggerSetup()
    logger.setup_logger("find_station_region")
    
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
            logger.info("=========================================")
            logger.info(f"{counter}-Site name: {name}")
            site_number = USGS_CONUS[USGS_CONUS.station_nm == name].site_no.values[0]
            logger.info(f"{counter}-Site number: {site_number}")
            # Find VPUID
            VPUID = find_VPUID(site_number)
            logger.info(f"{counter}-VPUID: {VPUID}")
            # Get coordinates
            x = USGS_CONUS[USGS_CONUS.site_no == site_number].geometry.x.values[0]
            y = USGS_CONUS[USGS_CONUS.site_no == site_number].geometry.y.values[0]
            logger.info(f"{counter}-Coordinates: {x:.2f}, {y:.2f}")
            ### find COUNTY_NAME in US_BOUNDARIES by using x and y
            COUNTY = US_BOUNDARIES[US_BOUNDARIES.geometry.contains(Point(x, y))].COUNTY_NAME.values[0]
            STATE = US_BOUNDARIES[US_BOUNDARIES.geometry.contains(Point(x, y))].STATE_NAME.values[0]
            logger.info(f"{counter}-State: {STATE}")
            logger.info(f"{counter}-County: {COUNTY}")
            logger.info("=========================================")
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
    search_term = "metal"
    df = find_station_region(search_term)