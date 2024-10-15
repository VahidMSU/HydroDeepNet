import glob
import os
import pandas as pd
def get_all_VPUIDs():
    path = "/data/SWATGenXApp/GenXAppData/NHDPlusData/NHDPlus_VPU_National/"
    ## get all file names ending with .zip
    files = glob.glob(f"{path}*.zip")

    VPUIDs = [os.path.basename(file).split('_')[2] for file in files]

    print(VPUIDs)
    return VPUIDs

def find_VPUID(station_no):
	CONUS_streamflow_data = pd.read_csv("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv", dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]