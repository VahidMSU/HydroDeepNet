import os
import pandas as pd

try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
    from SWATGenX.utils import get_all_VPUIDs
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths
    from utils import get_all_VPUIDs


def write_available_sites(VPUIDs, rewrite=False):
    """
    Writes available sites to a CSV file based on provided VPUIDs.

    Args:
        VPUIDs (list): List of VPUIDs to process.
        rewrite (bool): If True, overwrite the existing available sites file.

    Returns:
        pd.DataFrame: DataFrame containing all available stations.
    """
    if not os.path.exists(SWATGenXPaths.available_sites_path) or rewrite:
        all_stations = []
        for VPUID in VPUIDs:
            print(f"Processing VPUID: {VPUID}")
            streamflow_path = SWATGenXPaths.streamflow_path
            streamflow_metadata_path = f"{streamflow_path}/VPUID/{VPUID}/meta_{VPUID}.csv"

            if not os.path.exists(streamflow_metadata_path):
                print(f"File not found: {streamflow_metadata_path}")
                continue

            station_data = pd.read_csv(streamflow_metadata_path, dtype={'site_no': str, "first_huc": str})
            all_stations.append(station_data)
        print(f"all_stations: {all_stations}")
        all_stations = pd.concat(all_stations)
        all_stations['site_no'] = all_stations['site_no'].astype(str)  # Convert 'site_no' to object type
        all_stations.to_csv(SWATGenXPaths.available_sites_path, index=False)
    else:
        all_stations = pd.read_csv(SWATGenXPaths.available_sites_path, dtype={'site_no': str})

    return all_stations


def integrate_streamflow_data(usgs_data_base=None):
    """
    Integrates streamflow data from various sources and outputs a consolidated DataFrame.

    Args:
        usgs_data_base (str, optional): Base path for USGS data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing integrated streamflow data.
    """
    rewrite = True

    print("Integrating streamflow data")
    VPUIDs = get_all_VPUIDs()
    all_stations = write_available_sites(VPUIDs, rewrite=rewrite)

    fps = pd.read_csv(SWATGenXPaths.FPS_State_Territories, dtype={'SiteNumber': str})
    print(f"fps columns: {fps.columns}")
    fps_all_stations = pd.merge(all_stations, fps, left_on="site_no", right_on="SiteNumber", how="left")
    fps_all_stations = fps_all_stations.dropna(subset=["site_no"])
    fps_all_stations = fps_all_stations.fillna("---")

    # Round drainage area and GAP percent
    fps_all_stations['drainage_area_sqkm'] = fps_all_stations['drainage_area_sqkm'].round(2)
    fps_all_stations['GAP_percent'] = fps_all_stations['GAP_percent'].round(2)

    # Rename columns for clarity
    fps_all_stations = fps_all_stations.rename(columns={
        "GAP_percent": "Streamflow records gap (1999-2022) (%)",
        "drainage_area_sqkm": "Drainage area (sqkm)",
        "first_huc": "HUC12 id of the station",
        "list_of_huc12s": "HUC12 ids of the watershed",
        "number_of_streamflow_data": "Number of records",
        "total_expected_days": "Number of expected records (1999-2022)"
    })

    # Arrange columns
    fps_all_stations = fps_all_stations[['site_no', "Drainage area (sqkm)", "Number of expected records (1999-2022)",
                                          "Streamflow records gap (1999-2022) (%)", "SiteName", "Status", "SiteNumber",
                                          "USGSFunding", "HUC12 id of the station", "HUC12 ids of the watershed"]]

    fps_all_stations = fps_all_stations.drop(columns=["SiteNumber"])
    fps_all_stations = fps_all_stations.rename(columns={"site_no": "SiteNumber"})

    df_CONUS = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str, "first_huc": str})

    # Merge with CONUS data for latitude and longitude
    fps_all_stations = pd.merge(fps_all_stations, df_CONUS, left_on="SiteNumber", right_on="site_no", how="left")
    fps_all_stations = fps_all_stations.rename(columns={"dec_lat_va": "Latitude", "dec_long_va": "Longitude"})

    # Round latitude and longitude
    fps_all_stations['Latitude'] = fps_all_stations['Latitude'].round(2)
    fps_all_stations['Longitude'] = fps_all_stations['Longitude'].round(2)

    # Drop unnecessary columns
    fps_all_stations = fps_all_stations.drop(columns=["agency_cd", "station_nm", "site_tp_cd", "coord_acy_cd",
                                                       "dec_coord_datum_cd", "alt_va", "alt_acy_va", "alt_datum_cd"])

    print("Done")
    fps_all_stations.to_csv(SWATGenXPaths.FPS_all_stations, index=False)

    return fps_all_stations


if __name__ == "__main__":
    fps_all_stations = integrate_streamflow_data()
    print(fps_all_stations.head())
