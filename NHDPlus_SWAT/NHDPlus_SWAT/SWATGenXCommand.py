from multiprocessing import Process
import os
import pandas as pd
from functools import partial
from NHDPlus_SWAT.core import SWATGenXCore
import logging
import geopandas as gpd
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename="/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/SWATGenX.log")
def generate_huc12_list(HUC8, VPUID):
    path = f"/data/MyDataBase/SWATGenXAppData/NHDPlusData/SWATPlus_NHDPlus/{VPUID}/unzipped_NHDPlusVPU/"
    gdb = os.listdir(path)
    gdb = [g for g in gdb if g.endswith('.gdb')]
    path = os.path.join(path, gdb[0])
    huc12 = gpd.read_file(path, driver='FileGDB', layer='WBDHU12').to_crs('EPSG:4326')
    huc8 = gpd.read_file(path, driver='FileGDB', layer='WBDHU8').to_crs('EPSG:4326')
    ## intersect huc8 with huc12
    huc12 = gpd.overlay(huc12, huc8, how='intersection')
    huc12_grouped = huc12.groupby('HUC8')
    return {huc8: group['HUC12'].values for huc8, group in huc12_grouped}


def SWATGenXCommand(BASE_PATH, LEVEL, MAX_AREA, MIN_AREA, GAP_percent, landuse_product, landuse_epoch, ls_resolution, dem_resolution, station_name, MODEL_NAME, single_model, multiple_model_creation,target_VPUID):
	if LEVEL == "huc12":
		if single_model:
			USGS_streamflow_metadata = os.path.join(BASE_PATH,"USGS/streamflow_stations/CONUS/streamflow_stations_CONUS.csv")
			stations = pd.read_csv(USGS_streamflow_metadata, dtype={'site_no': str,'huc_cd': str})
			print(f"stations: {stations.huc_cd}")
			huc_cd = stations[stations.site_no == station_name].huc_cd.values[0]
			print(f"huc_cd: {huc_cd}")
			VPUID = huc_cd[:4]
			print(f"VPUID: {VPUID}")
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

			SWATGenXCore(station_name, BASE_PATH, VPUID, LEVEL, landuse_product, landuse_epoch, ls_resolution, dem_resolution,
								list_of_huc12s, MODEL_NAME)
			logging.info(f"station name path: {os.path.join(BASE_PATH,f'SWATplus_by_VPUID/{VPUID}/{LEVEL}/{station_name}/')}")

			return 	os.path.join(BASE_PATH,f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{station_name}/")

		elif multiple_model_creation:

			VPUID = target_VPUID

			streamflow_metadata = os.path.join(BASE_PATH,f"USGS/streamflow_stations/VPUID/{VPUID}/meta_{VPUID}.csv")
			streamflow_metadata = pd.read_csv(streamflow_metadata, dtype={'site_no': str})
			eligible_stations = streamflow_metadata[(streamflow_metadata.drainage_area_sqkm < MAX_AREA) & (streamflow_metadata.drainage_area_sqkm > MIN_AREA)]
			eligible_stations = eligible_stations[eligible_stations.GAP_percent < GAP_percent]
			import time
			print(f"eligible_stations: {eligible_stations.site_no}")
			if len(eligible_stations) == 0:
				logging.error(f"No eligible stations found for VPUID: {VPUID}")
				return None
			processes = []
			for site_no in eligible_stations.site_no:
				list_of_huc12s = eligible_stations[eligible_stations.site_no == site_no].list_of_huc12s.values[0]
				wrapped_SWATGenXCore = partial(SWATGenXCore, list_of_huc12s=list_of_huc12s, BASE_PATH=BASE_PATH, VPUID=VPUID, LEVEL=LEVEL, landuse_product=landuse_product, landuse_epoch=landuse_epoch, ls_resolution=ls_resolution, dem_resolution=dem_resolution)

				p = Process(target=wrapped_SWATGenXCore, args=(site_no,))
				p.start()
				processes.append(p)
				if len(processes) == 1:
					for p in processes:
						p.join()
					processes = []

			for p in processes:
				p.join()


	if LEVEL == "huc8":
		huc8_huc12_dict = generate_huc12_list(LEVEL, target_VPUID)
		### site_no is huc8

		for huc8, huc12s in huc8_huc12_dict.items():
			site_no = huc8
			VPUID = target_VPUID
			list_of_huc12s = huc12s
			SWATGenXCore(site_no, BASE_PATH, VPUID, LEVEL, landuse_product, landuse_epoch, ls_resolution, dem_resolution,
								list_of_huc12s, MODEL_NAME)
			return 	os.path.join(BASE_PATH,f"SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/")
