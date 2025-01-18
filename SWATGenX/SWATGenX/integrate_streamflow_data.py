import os
import pandas as pd
import glob
try:
	from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
	from SWATGenXConfigPars import SWATGenXPaths


def get_all_VPUIDs():
	NHDPlus_VPU_National_path = SWATGenXPaths.NHDPlus_VPU_National_path
	files = glob.glob(f"{NHDPlus_VPU_National_path}*.zip")
	return [os.path.basename(file).split('_')[2] for file in files]


def write_all_station_data(VPUIDs, usgs_data_base, rewrite=False):

	if not os.path.exists(os.path.join(usgs_data_base, "all_VPUIDs.csv")) or rewrite:
		all_stations = []
		for VPUID in VPUIDs:
			print(f"Processing VPUID: {VPUID}")
			#if VPUID[:2] != "04":
			#	continue
			streamflow_path = SWATGenXPaths.streamflow_path
			streamflow_metadata_path = f"{streamflow_path}/VPUID/{VPUID}/meta_{VPUID}.csv"
		
			if not os.path.exists(streamflow_metadata_path):
				print(f"File not found: {streamflow_metadata_path}")
				continue
				
			station_data = pd.read_csv(streamflow_metadata_path, dtype={'site_no': str, "first_huc": str})	

			all_stations.append(station_data)

		all_stations = pd.concat(all_stations)
		all_stations['site_no'] = all_stations['site_no'].astype(str)  # Convert 'site_no' to object type
		all_stations.to_csv(os.path.join(usgs_data_base, "all_VPUIDs.csv"), index=False)
	else:
		all_stations = pd.read_csv(os.path.join(usgs_data_base, "all_VPUIDs.csv"), dtype={'site_no': str})

	return all_stations

def integrate_streamflow_data(usgs_data_base):

#	if not rewrite and os.path.exists(os.path.join(usgs_data_base, "FPS_all_stations.csv")):
#		print("Reading existing streamflow data")
#		return pd.read_csv(os.path.join(usgs_data_base, "FPS_all_stations.csv"), dtype={'site_no': str})
	
	rewrite=False

	print("Integrating streamflow data")
	VPUIDs = get_all_VPUIDs()
	all_stations = []
	
	all_stations = write_all_station_data(VPUIDs, usgs_data_base, rewrite=rewrite)
	
	fps = pd.read_csv(os.path.join(usgs_data_base, "FPS_States_and_Territories.csv"), skiprows=1, dtype={'SiteNumber': str})	

	fps_all_stations = pd.merge(all_stations, fps, left_on="site_no", right_on="SiteNumber", how="left")
	fps_all_stations = fps_all_stations.dropna(subset=["site_no"])
	fps_all_stations = fps_all_stations.fillna("---")

	### we need the column drainage_area_sqkm to be float with 2 decimal points
	fps_all_stations['drainage_area_sqkm'] = fps_all_stations['drainage_area_sqkm'].round(2)
	fps_all_stations['GAP_percent'] = fps_all_stations['GAP_percent'].round(2)
	fps_all_stations = fps_all_stations.rename(columns = {"GAP_percent": "Streamflow records gap (1999-2022) (%)"})
	fps_all_stations = fps_all_stations.rename(columns = {"drainage_area_sqkm": "Drainage area (sqkm)"})
	fps_all_stations = fps_all_stations.rename(columns = {"first_huc": "HUC12 id of the station", "list_of_huc12s": "HUC12 ids of the watershed"})
	fps_all_stations = fps_all_stations.rename(columns = {"number_of_streamflow_data": "Number of records", "total_expected_days": "Number of expected records (1999-2022)"})
	### now arrange the columns

	fps_all_stations = fps_all_stations[['site_no',"Drainage area (sqkm)","Number of expected records (1999-2022)",
                                    "Streamflow records gap (1999-2022) (%)", "SiteName", "Status", "SiteNumber", "USGSFunding", 
                                    "HUC12 id of the station", "HUC12 ids of the watershed",
                                    ]]

	fps_all_stations = fps_all_stations.drop(columns = ["SiteNumber"])
	fps_all_stations = fps_all_stations.rename(columns = {"site_no": "SiteNumber"})
		
	USGS_CONUS_path = os.path.join(usgs_data_base, "streamflow_stations/CONUS/streamflow_stations_CONUS.csv")	
	df_CONUS = pd.read_csv(USGS_CONUS_path, dtype={'site_no': str, "first_huc": str})
	#agency_cd,site_no,station_nm,site_tp_cd,dec_lat_va,dec_long_va,coord_acy_cd,dec_coord_datum_cd,alt_va,alt_acy_va,alt_datum_cd,huc_cd
	### now merge the fps_all_stations with df_CONUS to get the decimal latitude and longitude
	
	fps_all_stations = pd.merge(fps_all_stations, df_CONUS, left_on="SiteNumber", right_on="site_no" ,how="left")
	
	fps_all_stations = fps_all_stations.rename(columns = {"dec_lat_va": "Latitude", "dec_long_va": "Longitude"})
	# with two decimal degrees
	fps_all_stations['Latitude'] = fps_all_stations['Latitude'].round(2)
	fps_all_stations['Longitude'] = fps_all_stations['Longitude'].round(2)
	
	fps_all_stations = fps_all_stations.drop(columns = ["agency_cd", "station_nm", "site_tp_cd", "coord_acy_cd", "dec_coord_datum_cd", "alt_va", "alt_acy_va", "alt_datum_cd"])
	
	print("Done")
	fps_all_stations.to_csv(os.path.join(usgs_data_base, "FPS_all_stations.csv"), index=False)
	
	return fps_all_stations

if __name__ == "__main__":
	usgs_data_base = "/data/SWATGenXApp/GenXAppData/USGS/"
	fps_all_stations = integrate_streamflow_data(usgs_data_base)
	print(fps_all_stations.head())