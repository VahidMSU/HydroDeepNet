from SWATGenX.USGS_monitoring_extraction import get_streamflow_stations_for_state, organizing_stations_into_CONUS, organizing_CONUS_by_VPUID
import os

if __name__ == "__main__":
    """ These operations create shapefile containing the location and a csv file containing the information of USGS streamflow stations..
    ...We first retrieve streamflow data (csv) from USGS for each state, then combine them into a single shapefile for the CONUS, and finally organize them by VPUID.
    We only need to run this script once to get the data.
    """

    state_cds = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma",
        "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
        "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "pr", "vi", "gu", "as", "mp"]

    base_directory = "/data/SWATGenXApp/GenXAppData/USGS/"
    os.makedirs("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/", exist_ok=True)
    os.makedirs("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/state/", exist_ok=True)
    os.makedirs("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/", exist_ok=True)
    os.makedirs("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/VPUID/", exist_ok=True)

    get_streamflow_stations_for_state(base_directory, state_cds)
    organizing_stations_into_CONUS(base_directory, state_cds)
    organizing_CONUS_by_VPUID(base_directory)
