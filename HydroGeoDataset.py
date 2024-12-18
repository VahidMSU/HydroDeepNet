from HydroGeoDataset.HydroGeoDataset.core import DataImporter

if __name__ == '__main__':





#################################################

	# get mask
	#mask = importer.get_mask()


	# Import annual precipitation PRISM data for a specific time period
	#annual_rainfall = importer.import_transient_data(range(1990, 2022), "ppt")   ## mm
	#average_daily_recharge =importer.import_transient_data(range(1990, 2022), "recharge")  ## mm/day

	#import numerical and categorical datasets
		# import PFAS data #
	extract_pfas_data = False
	if extract_pfas_data:
		config = {
			"RESOLUTION": 250,
			"PFAS": "PFOS"}   ### othr PFAS: PFHxS, PFOA, PFNA, PFDA, PFOS
		importer = DataImporter(config)

		pfas_max, pfas_mean, pfas_std = importer.import_pfas_data()

	##########################################################################
	# extract information for point locations
	extract_point_data = False
	if extract_point_data:
		config = {
			"RESOLUTION": 250}

		input_path = "/data2/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites.geojson"
		output_path = "/data/MyDataBase/test.pkl"
		importer = DataImporter(config)
		gdf = importer.extract_features(input_path)
		gdf.to_pickle(output_path)
		gdf.to_file(output_path.replace(".pkl", ".shp"))
	
	single_location_extraction = True
	if single_location_extraction:
		### a random location within Michigan LP
		config = {
			"RESOLUTION": 250}
		importer = DataImporter(config)
		lat, lon = 42.0, -84.0
		gdf = importer.extract_features(single_location=(lat, lon))

		print(f"Extracted features: {gdf.columns}")

	###########################################################################

	# Plot the shapefile
	#gdf.plot()
	#plt.savefig("input_figs/P_locations_rasters_30m.png")

	# import snowdas data
	#melt_rate = importer.extract_snowdas_data(snowdas_var='snow_layer_thickness', year = 2015)  #'melt_rate', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate'. data range from 2004 to 2019


	#config = {
	#	"RESOLUTION": 250,
	#	"huc8": "4060105",
		#"video": True,
	#}
	#importer = DataImporter(config)


	#gw_3d_ds = importer.gw_3d_ds(start_year=2020, end_year=2021)

	#gw_station_data = importer.gw_stations_ds(start_year=1990, end_year=2021)
	# print detail of one station
	# print the features of the station
	#print(f"numerical feature: {gw_station_data['421332085401901_1609_389']['numerical_feature']}")
	#print(f"categorical feature: {gw_station_data['421332085401901_1609_389']['categorical_feature']}")
	# print(f"head: {gw_station_data['421332085401901_1609_389']}")






	# import training and testing dataset for 3D CNN-LSTM
