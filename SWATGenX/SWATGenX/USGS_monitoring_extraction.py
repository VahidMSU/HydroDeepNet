import pandas as pd
import geopandas as gpd
import os
import requests
from shapely.geometry import Point
import warnings
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
def get_streamflow_stations_for_state(state_cds):


    for state_cd in state_cds:
        print(f"Getting streamflow stations for {state_cd}")
        """ Get the streamflow stations for each state and save it to a csv file, plus a shapefile"""
        try:
            url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={state_cd}&siteStatus=all&siteType=ST&hasDataTypeCd=dv"
            locations_file = SWATGenXPaths.USGS_CONUS_stations_path
            os.makedirs(os.path.dirname(locations_file), exist_ok=True)
            shapefile = SWATGenXPaths.USGS_CONUS_stations_shape_path
            if os.path.exists(locations_file):
                print(f"Data for {state_cd} already exists")
                continue
            response = requests.get(url)
            if response.status_code == 200:
                data = response.text
                # save the data to a file
                with open(locations_file, "w") as f:
                    f.write(data)
                print(f"Data saved to {locations_file}")
            else:
                print("Error retrieving data from the URL")

            # read the data into a pandas dataframe
            df = pd.read_csv(locations_file, skiprows=29, delimiter="\t", dtype={"site_no": str})
            # drop the first row
            df = df.drop(0)

            # save the data to a shapefile
            # first create the geometry column
            geometry = df.apply(lambda row: Point(row.dec_long_va, row.dec_lat_va), axis=1)
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

            warnings.filterwarnings("ignore", message="Column names longer than 10 characters will be truncated when saved to ESRI Shapefile")
            gdf.to_file(shapefile)
        except Exception as e:
            print(f"Error: {e}")
            continue


### now read all csv files and make them one file
def organizing_stations_into_CONUS(state_cds):
    """ Read all the csv files and make them one file to represent CONUS"""
    print("Organizing stations into CONUS")
    all_data = pd.DataFrame()
    for _ in state_cds:
        locations_file = SWATGenXPaths.USGS_CONUS_stations_path
        if os.path.exists(locations_file):
            df = pd.read_csv(locations_file, skiprows=29, delimiter="\t", dtype={"site_no": str})
            df = df.drop(0)
            all_data = pd.concat([all_data, df])

    ## now save the in CONUS
    os.makedirs(os.path.dirname(SWATGenXPaths.USGS_CONUS_stations_path), exist_ok=True)
    all_data.to_csv(SWATGenXPaths.USGS_CONUS_stations_path, index=False)
    ## make the shapefile
    all_data['geometry'] = all_data.apply(lambda row: Point(row.dec_long_va, row.dec_lat_va), axis=1)
    gdf = gpd.GeoDataFrame(all_data, geometry='geometry', crs="EPSG:4326")
    gdf.to_file(SWATGenXPaths.USGS_CONUS_stations_shape_path)

def organizing_CONUS_by_VPUID():
    """ organize the CONUS data based on VPUID """
    print("Organizing CONUS data by VPUID")
    all_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={"site_no": str})
    ## convert huc_cd column to string
    all_data['huc_cd'] = all_data['huc_cd'].astype(str)
    ## filter out NaN values
    ## make sure huc_cd is string
    all_data['VPUID'] = ''

    all_data
    for i in range(len(all_data)):
        try:
            all_data.loc[i, 'VPUID'] = str(all_data.loc[i, 'huc_cd'])[:4]
        except Exception as e:
            print(f"Error: {e}")
    ## read the VPUID data
    ## VPUID is the first 4 letters of huc_cd column

    VPUIDs = all_data['VPUID'].unique()
    print(f"Number of VPUIDs: {len(VPUIDs)}")
    print(f"VPUIDs: {VPUIDs}")
    for VPUID in VPUIDs:
        print(f"Organizing CONUS by VPUID: {VPUID}")
        df = all_data[all_data['VPUID'] == VPUID]

        os.makedirs(f"{SWATGenXPaths.streamflow_path}/VPUID/{VPUID}", exist_ok=True)
        df.to_csv(SWATGenXPaths.USGS_CONUS_stations_path, index=False)
        geometry = df.apply(lambda row: Point(row.dec_long_va, row.dec_lat_va), axis=1).copy()
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        warnings.filterwarnings("ignore", message="Column names longer than 10 characters will be truncated when saved to ESRI Shapefile")
        gdf.to_file(f"{SWATGenXPaths.streamflow_path}/VPUID/{VPUID}/streamflow_stations_{VPUID}.shp")
