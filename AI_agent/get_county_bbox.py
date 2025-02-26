def get_bounding_box(county, state):
    import geopandas 
    import numpy as np

    path = "/data/SWATGenXApp/GenXAppData/USGS/GovernmentUnits_National_GDB/GovernmentUnits_National_GDB.gdb/"

    try:
        gdf = geopandas.read_file(path, layer="GU_CountyOrEquivalent")
        # Case-insensitive search
        county_shape = gdf[
            (gdf["STATE_NAME"].str.lower() == state.lower()) & 
            (gdf["COUNTY_NAME"].str.lower() == county.lower())
        ]
        
        if county_shape.empty:
            print(f"Could not find {county} County in {state}")
            return None, None, None, None
            
        county_shape = county_shape.to_crs("EPSG:4326")
        bbox = county_shape.total_bounds.tolist()
        min_lon, min_lat, max_lon, max_lat = bbox

        if any(map(np.isnan, [min_lon, min_lat, max_lon, max_lat])):
            print(f"Invalid coordinates found for {county} County, {state}")
            return None, None, None, None

        return min_lon, min_lat, max_lon, max_lat

    except Exception as e:
        print(f"Error getting bounding box: {e}")
        return None, None, None, None

if __name__ == "__main__":
    min_lon, min_lat, max_lon, max_lat = get_bounding_box("Mecosta", "Michigan")
    print(f"Bounding box for Ingham County, Michigan (EPSG:4326): {min_lon, min_lat, max_lon, max_lat}")
