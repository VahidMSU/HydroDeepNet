import pandas as pd
from ModelProcessing.logging_utils import get_logger

# Get a logger instance with a name that identifies this module
logger = get_logger(__name__)

def find_VPUID(station_no):
	CONUS_streamflow_data = pd.read_csv("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv", dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]

if __name__ == "__main__":
	station_no = "01343060"
	VPUID = find_VPUID(station_no)
	logger.info(f"VPUID for station {station_no} is {VPUID}")
	# VPUID for station 01343060 is 0204