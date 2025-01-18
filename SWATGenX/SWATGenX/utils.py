import os
import pandas as pd
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenXConfigPars import SWATGenXPaths

def get_all_VPUIDs():
    import glob
    extracted_nhd_swatplus_path = SWATGenXPaths.extracted_nhd_swatplus_path
    files = glob.glob(f"{extracted_nhd_swatplus_path}*.zip")
    VPUIDs = [os.path.basename(file).split('_')[2] for file in files]
    print(VPUIDs)
    return VPUIDs

def find_VPUID(station_no):
    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    return CONUS_streamflow_data[
        CONUS_streamflow_data.site_no == station_no
    ].huc_cd.values[0][:4]