import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
def find_VPUID(station_no):
	CONUS_streamflow_data = pd.read_csv("/data/MyDataBase/CIWRE-BAE/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv", dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]

if __name__ == "__main__":
	station_no = "01343060"
	VPUID = find_VPUID(station_no)
	logging.info(f"VPUID for station {station_no} is {VPUID}")
	# VPUID for station 01343060 is 0204