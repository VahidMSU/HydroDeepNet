from dataclasses import dataclass
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
def fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MODEL_NAME):
    paths = SWATGenXPaths()

    model_base = paths.construct_path(paths.swatgenx_outlet_path, VPUID, LEVEL, NAME)
    swatplus_stations_shp = Path(paths.construct_path(model_base, "streamflow_data", "stations.shp"))

    meta_data_path = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"meta_{VPUID}.csv")
    streamflow_stations_shp = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"streamflow_stations_{VPUID}.shp")
    swatplus_lsus2_shp = paths.construct_path(model_base, MODEL_NAME, "Watershed", "Shapes", "lsus2.shp")
    target_path = paths.construct_path(model_base, "streamflow_data")

    # Read data
    stations = gpd.read_file(streamflow_stations_shp).to_crs("EPSG:4326")
    swatplus_lsus2 = gpd.read_file(swatplus_lsus2_shp).to_crs("EPSG:4326")
    meta_data = pd.read_csv(meta_data_path, dtype={"site_no": str})

    # Create directories
    swatplus_stations_shp.parent.mkdir(parents=True, exist_ok=True)

    # Spatial join
    subbasins_stations = gpd.sjoin(stations, swatplus_lsus2, how="inner", predicate="intersects")
    subbasins_stations[["site_no", "geometry"]].to_file(swatplus_stations_shp)

    # Copy streamflow data
    for channel, site_no in zip(subbasins_stations["Channel"], subbasins_stations["site_no"]):
        source_jpeg = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"streamflow_{site_no}.jpeg")
        if os.path.exists(source_jpeg):
            shutil.copy(source_jpeg, target_path)
            print(f"Streamflow data for {site_no} is copied to {target_path}")
        else:
            print(f"JPEG file for site {site_no} does not exist.")

        source_record_jpeg = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"streamflow_record_{site_no}.jpeg")
        if os.path.exists(source_record_jpeg):
            shutil.copy(source_record_jpeg, target_path)
        else:
            print(f"Record JPEG for site {site_no} does not exist.")

        source_csv = paths.construct_path(paths.streamflow_path, "VPUID", VPUID, f"streamflow_{site_no}.csv")
        target_csv = paths.construct_path(target_path, f"{channel}_{site_no}.csv")
        if os.path.exists(source_csv):
            shutil.copy(source_csv, target_csv)
        else:
            print(f"CSV file for site {site_no} does not exist.")

if __name__ == "__main__":
    VPUID = "0206"
    LEVEL = "huc12"
    NAME = "01583570"
    MODEL_NAME = "SWAT_MODEL"
    fetch_streamflow_for_watershed(VPUID, LEVEL, NAME, MODEL_NAME)
