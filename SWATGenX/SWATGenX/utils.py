import os
import pandas as pd
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except Exception:
    from SWATGenXConfigPars import SWATGenXPaths

def get_all_VPUIDs():
    VPUIDs_path = f"{SWATGenXPaths.DEM_path}/VPUID/"
    VPUIDs = os.listdir(VPUIDs_path)
    VPUIDs = [VPUID for VPUID in VPUIDs if os.path.isdir(os.path.join(VPUIDs_path, VPUID))]
    return VPUIDs

def find_VPUID(station_no):
    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    return CONUS_streamflow_data[
        CONUS_streamflow_data.site_no == station_no
    ].huc_cd.values[0][:4]




def return_list_of_huc12s(station_name):
    """
    Returns a list of HUC12s for a given station name and maximum drainage area.

    This method checks the drainage area of the specified station and retrieves eligible HUC12s based on the maximum area criteria.

    Args:
        station_name (str): The name of the station to retrieve HUC12s for.
        max_area (float): The maximum drainage area allowed.

    Returns:
        tuple: A tuple containing the list of HUC12s and the corresponding VPUID.
    """
    vpuid = find_VPUID(station_name)
    streamflow_metadata = f"{SWATGenXPaths.streamflow_path}/VPUID/{vpuid}/meta_{vpuid}.csv"
    streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})

    list_of_huc12s = streamflow_metadata[streamflow_metadata.site_no == station_name].list_of_huc12s.values[0]
    print(f"Station name: {station_name}, VPUID: {vpuid}")
    list_of_huc12s = {int(huc12.strip("'")) for huc12 in list_of_huc12s[1:-1].split(", ")}
    list_of_huc12s = [f"{huc12:012d}" for huc12 in list_of_huc12s]
    
    return list_of_huc12s, vpuid



