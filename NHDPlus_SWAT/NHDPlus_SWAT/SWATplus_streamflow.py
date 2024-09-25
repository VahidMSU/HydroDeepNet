import pandas as pd
import geopandas as gpd
from pathlib import Path
import distutils.dir_util
import distutils.file_util
import os
import matplotlib.pyplot as plt


def fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MDOEL_NAME):
    base = "/data/MyDataBase/SWATGenXAppData/USGS/streamflow_stations/"
    model_base = f"/data/MyDataBase/SWATGenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/"
    swatplus_stations_shp = Path(model_base,"streamflow_data/stations.shp")

    meta_data = os.path.join(base,f"VPUID/{VPUID}/meta_{VPUID}.csv")
    streamflow_stations_shp = os.path.join(base,f"VPUID/{VPUID}/streamflow_stations_{VPUID}.shp")
    swatplus_lsus2_shp = os.path.join(model_base,f"{MDOEL_NAME}/Watershed/Shapes/lsus2.shp")
    swatplus_stations_shp = os.path.join(model_base,"streamflow_data/stations.shp")
    target_path = os.path.join(model_base,"streamflow_data/")

    # Read data
    stations = gpd.read_file(streamflow_stations_shp).to_crs("EPSG:4326")
    swatplus_lsus2 = gpd.read_file(swatplus_lsus2_shp).to_crs("EPSG:4326")
    meta_data = pd.read_csv(meta_data, dtype={"site_no": str})

    # Create directories
    Path(swatplus_stations_shp).parent.mkdir(parents=True, exist_ok=True)

    # Spatial join
    subbasins_stations = gpd.sjoin(stations, swatplus_lsus2, how="inner", predicate="intersects")
    subbasins_stations[['site_no','geometry']].to_file(swatplus_stations_shp)

    # Copy streamflow data
    for channel, site_no in zip(subbasins_stations["Channel"], subbasins_stations["site_no"]):
        try:
            source_jpeg = os.path.join(base,f"VPUID/{VPUID}/streamflow_{site_no}.jpeg")
            distutils.file_util.copy_file(source_jpeg, target_path)
            print(f"Streamflow data for {site_no} is copied to {target_path}")
        except Exception as e:
            print(f"Error. site {site_no}: {e}")
        try:
            source_record_jpeg = os.path.join(base,f"VPUID/{VPUID}/streamflow_record_{site_no}.jpeg")
            distutils.file_util.copy_file(source_record_jpeg, target_path)
        except Exception as e:
            print(f"Error. site {site_no}: {e}")
        try:
            source_csv = os.path.join(base,f"VPUID/{VPUID}/streamflow_{site_no}.csv")
            target_csv = os.path.join(target_path , f"{channel}_{site_no}.csv")
            distutils.file_util.copy_file(source_csv, target_csv)
        except Exception as e:
            print(f"Error. site {site_no}: {e}")

if __name__ == "__main__":
    VPUID = "0407"
    LEVEL = "huc12"
    NAME = "04127997"
    MDOEL_NAME = "SWAT_MODEL"
    fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MDOEL_NAME)
