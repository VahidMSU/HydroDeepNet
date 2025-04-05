import sys 
sys.path.append('/data/SWATGenXApp/codes/SWATGenX/')
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_station_geometries():
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    import pandas as pd
    import geopandas as gpd
    df = pd.read_csv(SWATGenXPaths.FPS_all_stations, dtype={'SiteNumber': str})
    print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging
    ## check the columns
    import time 
    time.sleep(500)
    print(f"Columns in DataFrame: {df.columns.tolist()}")  # Debugging
    ## limit to CONUS based on Latitude and Longitude
    df = df[(df['Latitude'] >= 24.396308) & (df['Latitude'] <= 49.384358) & 
            (df['Longitude'] >= -125.0) & (df['Longitude'] <= -66.93457)]
    ## limit the drainage size to maximum 3000 sqk
    df = df[df['Drainage area (sqkm)'] <= 3000]
    ## convert to geodataframe
    from shapely.geometry import Point
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    ## convert to geodataframe
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    ## set coordinate system
    gdf.crs = "EPSG:4326"
    ## convert to EPSG:4326
    gdf = gdf.to_crs(epsg=4326)
    ## change the base path ownership to rafieiva
    import os 
    
    # Include all important columns in the GeoJSON file
    columns_to_keep = [
        'SiteName', 'SiteNumber', 'Latitude', 'Longitude', 'site_no', 
        'Drainage area (sqkm)', 'Number of expected records (1999-2022)',
        'Streamflow records gap (1999-2022) (%)', 'Status', 'USGSFunding', 
        'HUC12 id of the station', 'HUC12 ids of the watershed', 'huc_cd',
        'geometry'
    ]
    
    # Ensure all columns exist before filtering
    available_columns = [col for col in columns_to_keep if col in gdf.columns]
    gdf = gdf[available_columns]

    gdf.to_file(SWATGenXPaths.FPS_CONUS_stations, driver='GeoJSON')
    ## read the GeoJSON file
    gdf = gpd.read_file(SWATGenXPaths.FPS_CONUS_stations, driver='GeoJSON')
    ## set coordinate system


    print(f"Columns in GeoDataFrame: {gdf.columns.tolist()}")  # Debugging
    ## number of stations
    print(f"Number of stations: {len(gdf)}")

    

    return gdf


if __name__ == "__main__":
    # Load the station geometries
    gdf = load_station_geometries()

