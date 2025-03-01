
from agent import AgentConfig

def get_bounding_box(county, state):
    import geopandas 
    import numpy as np

    # Add validation for county and state inputs
    if county is None or state is None:
        print(f"Error: County or state cannot be None. Received county={county}, state={state}")
        return None, None, None, None
        
    path = AgentConfig.USGS_governmental_path

    try:
        gdf = geopandas.read_file(path, layer="GU_CountyOrEquivalent")
        # Case-insensitive search
        matching_counties = gdf[
            (gdf["STATE_NAME"].str.lower() == state.lower()) & 
            (gdf["COUNTY_NAME"].str.lower() == county.lower())
        ]
        
        if matching_counties.empty:
            available_counties = gdf[gdf["STATE_NAME"].str.lower() == state.lower()]["COUNTY_NAME"].tolist()
            print(f"Error: '{county}' not found in {state}.")
            print(f"Available counties in {state}: {', '.join(sorted(available_counties))}")
            return None, None, None, None
            
        county_shape = matching_counties.to_crs("EPSG:4326")
        bbox = county_shape.total_bounds.tolist()
        min_lon, min_lat, max_lon, max_lat = bbox

        return min_lon, min_lat, max_lon, max_lat

    except Exception as e:
        print(f"Error accessing county database: {e}")
        return None, None, None, None

if __name__ == "__main__":
    min_lon, min_lat, max_lon, max_lat = get_bounding_box("Mecosta", "Michigan")
    print(f"Bounding box for Ingham County, Michigan (EPSG:4326): {min_lon, min_lat, max_lon, max_lat}")
