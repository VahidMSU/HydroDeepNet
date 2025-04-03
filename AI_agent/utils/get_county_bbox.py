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
    counties = gdf[gdf["COUNTY_NAME"] == county_name]
    if counties.empty:
        logger.warning(f"No counties found with name {county_name}")
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
if __name__ == "__main__":
    min_lon = get_bounding_box("Ingham", "MI")
    print(f"Bounding box for Mecosta County, MI: {min_lon}")
