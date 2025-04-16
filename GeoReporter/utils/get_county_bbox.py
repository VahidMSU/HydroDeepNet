import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_bounding_box(county_name=None, state_code=None, county_fips=None):
    DEFAULT_COUNTY_DATA_PATH = "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"

    # Path to county boundaries data file
    import geopandas as gpd
    ## read layers
    gdf = gpd.read_file(
        DEFAULT_COUNTY_DATA_PATH,
        layer="GU_CountyOrEquivalent"
    )
    
    # Filter by county name and state code if provided
    if county_name and state_code:
        counties = gdf[(gdf["COUNTY_NAME"] == county_name) & (gdf["STATE_NAME"] == state_code)]
    elif county_name:
        counties = gdf[gdf["COUNTY_NAME"] == county_name]
    elif county_fips:
        counties = gdf[gdf["FIPS_CODE"] == county_fips]
    else:
        logger.warning("No filtering criteria provided (county_name, state_code, or county_fips)")
        return None
    
    if counties.empty:
        logger.warning(f"No counties found with the provided criteria (name: {county_name}, state: {state_code}, fips: {county_fips})")
        return None
    else:
        # Get the first matching county
        county = counties.iloc[0]

        # Get the bounding box coordinates
        min_lon, min_lat, max_lon, max_lat = county.geometry.bounds

        # Create a dictionary with the bounding box coordinates
        bounding_box = {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }

        return bounding_box


def get_counties_by_state(state_code):
    """Return a list of county names for the given state code"""
    DEFAULT_COUNTY_DATA_PATH = "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"
    
    import geopandas as gpd
    gdf = gpd.read_file(
        DEFAULT_COUNTY_DATA_PATH,
        layer="GU_CountyOrEquivalent"
    )
    
    if state_code:
        counties = gdf[gdf["STATE_NAME"] == state_code]
        if counties.empty:
            logger.warning(f"No counties found for state {state_code}")
            return []
        else:
            return counties["COUNTY_NAME"].tolist()
    else:
        logger.warning("No state code provided")
        return []


def get_state_codes():
    """Return a list of all state codes"""
    DEFAULT_COUNTY_DATA_PATH = "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb"
    
    import geopandas as gpd
    gdf = gpd.read_file(
        DEFAULT_COUNTY_DATA_PATH,
        layer="GU_CountyOrEquivalent"
    )
    
    return sorted(gdf["STATE_NAME"].unique().tolist())


if __name__ == "__main__":
    bbox = get_bounding_box("Ingham", "Michigan")
    print(f"Bounding box for Ingham County, MI: {bbox}")
    
    # Example of getting counties for a state
    counties = get_counties_by_state("Michigan")
    print(f"Counties in Michigan: {counties[:5]}... (showing first 5)")
    
    # Example of getting all state codes
    states = get_state_codes()
    print(f"Available states: {states}")
