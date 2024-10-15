from multiprocessing import Process
import os
import pandas as pd
from functools import partial
from SWATGenX.core import SWATGenXCore
import logging
import geopandas as gpd


def find_VPUID(station_no, LEVEL="huc12"):
	if LEVEL == "huc8":
		return f"0{int(station_no)[:3]}"

	CONUS_streamflow_data = pd.read_csv("/data/SWATGenXApp/GenXAppData/USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv", dtype={'site_no': str,'huc_cd': str})
	return CONUS_streamflow_data[
		CONUS_streamflow_data.site_no == station_no
	].huc_cd.values[0][:4]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename="/data/SWATGenXApp/codes/SWATGenX.log")
def generate_huc12_list(HUC8, VPUID):

	path = f"/data/SWATGenXApp/GenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
	gdb = os.listdir(path)
	gdb = [g for g in gdb if g.endswith('.gdb')]
	path = os.path.join(path, gdb[0])
	huc12 = gpd.read_file(path, driver='FileGDB', layer='WBDHU12').to_crs('EPSG:4326')
	huc8 = gpd.read_file(path, driver='FileGDB', layer='WBDHU8').to_crs('EPSG:4326')
	## intersect huc8 with huc12
	huc12 = gpd.overlay(huc12, huc8, how='intersection')
	## make the huc8 to HUC8 if it is not
	if "huc8" in huc12.columns:
		huc12.rename(columns={"huc8": "HUC8"}, inplace=True)
		## make sure its 8 digits and string
		huc12['HUC8'] = huc12[huc12.HUC8 == HUC8].HUC8.astype(str).str.zfill(8)
	huc12 = huc12[huc12.HUC8 == HUC8]
	
	if "huc12" in huc12.columns:
		huc12.rename(columns={"huc12": "HUC12"}, inplace=True)

	huc12_grouped = huc12.groupby('HUC8')
	return {huc8: group['HUC12'].values for huc8, group in huc12_grouped}


def return_list_of_huc12s(BASE_PATH, station_name, MAX_AREA):
	VPUID = find_VPUID(station_name)	
	streamflow_metadata = os.path.join(BASE_PATH,f"USGS/streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv")
	streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})

	drainage_area = streamflow_metadata[streamflow_metadata.site_no == station_name].drainage_area_sqkm.values[0]

	eligible_stations = streamflow_metadata[streamflow_metadata.drainage_area_sqkm < MAX_AREA]
	if len(eligible_stations) == 0:
		logging.error(f"Station {station_name} does not meet the maximum drainage area criteria: {MAX_AREA} sqkm")
		return None

	if len(eligible_stations[eligible_stations.site_no == station_name]) == 0:
		logging.error(f"Station {drainage_area} sqkm is greater than the maximum drainage area criteria: {MAX_AREA} sqkm")
		return None

	list_of_huc12s = eligible_stations[eligible_stations.site_no == station_name].list_of_huc12s.values[0]
	print(f"station name : {station_name}, VPUIID: {VPUID}")

	return list_of_huc12s, VPUID


def SWATGenXCommand(swatgenx_config):

	LEVEL = swatgenx_config.get("LEVEL")
	MAX_AREA = swatgenx_config.get("MAX_AREA")
	MIN_AREA = swatgenx_config.get("MIN_AREA")
	GAP_percent = swatgenx_config.get("GAP_percent")
	single_model = swatgenx_config.get("single_model", True)

	if LEVEL == "huc12":
		logging.info(f'LEVEL: {LEVEL}, station_name: {swatgenx_config.get("station_name")}, single_model: {single_model}')
		if single_model:

			station_name = swatgenx_config.get("station_name")
			list_of_huc12s, VPUID = return_list_of_huc12s(swatgenx_config.get("BASE_PATH"), swatgenx_config.get("station_name"), MAX_AREA)

			# Correct way to update the dictionary
			swatgenx_config.update(
				{
					"site_no": station_name,
					"VPUID": VPUID,
					"LEVEL": LEVEL,
					"list_of_huc12s": list_of_huc12s,
				}
			)
			SWATGenXCore(swatgenx_config)

			return os.path.join(swatgenx_config.get("BASE_PATH"), f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{station_name}/")

	elif LEVEL == "huc4":
		VPUID = swatgenx_config.get("target_VPUID")
		streamflow_metadata = os.path.join(swatgenx_config.get("BASE_PATH"), f"USGS/streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv")
		streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})
		eligible_stations = streamflow_metadata[(streamflow_metadata.drainage_area_sqkm < MAX_AREA) & (streamflow_metadata.drainage_area_sqkm > MIN_AREA)]
		eligible_stations = eligible_stations[eligible_stations.GAP_percent < GAP_percent]
		print(f"eligible_stations: {eligible_stations.site_no}")
		if len(eligible_stations) == 0:
			logging.error(f"No eligible stations found for VPUID: {VPUID}")
			return None
		processes = []
		for site_no in eligible_stations.site_no:

			# Correct way to update the dictionary
			swatgenx_config.update(
				{
					"site_no": site_no,
					"VPUID": VPUID,
					"list_of_huc12s": eligible_stations[eligible_stations.site_no == site_no].list_of_huc12s.values[0],
				}
			)
			wrapped_SWATGenXCore = partial(SWATGenXCore, config=swatgenx_config)

			p = Process(target=wrapped_SWATGenXCore, args=(site_no,))
			p.start()
			processes.append(p)
			if len(processes) == 1:
				for p in processes:
					p.join()
				processes = []

		for p in processes:
			p.join()

	elif LEVEL == "huc8":
		VPUID = swatgenx_config.get("station_name")[:4]
		HUC8_NAME = swatgenx_config.get("station_name")
		list_of_huc12s = generate_huc12_list(HUC8_NAME, VPUID)
		# Correct way to update the dictionary
		swatgenx_config.update(
			{
				"site_no": HUC8_NAME,
				"VPUID": VPUID,
				"LEVEL": LEVEL,
				"list_of_huc12s": list_of_huc12s,
			}
		)

		SWATGenXCore(swatgenx_config)
		return os.path.join(swatgenx_config.get("BASE_PATH"), f"SWATplus_by_VPUID/{swatgenx_config.get('VPUID')}/{LEVEL}/{HUC8_NAME}/")
