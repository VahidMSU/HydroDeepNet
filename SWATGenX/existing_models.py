import os
def check_existing_models(station_name):
	path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/"
	VPUIDs = os.listdir(path)
	existing_models = []
	for VPUID in VPUIDs:
		# now find model inside huc12 directory
		huc12_path = os.path.join(path, VPUID, "huc12")
		models = os.listdir(huc12_path)
		existing_models.extend(os.path.join(huc12_path, model) for model in models)


	for model in existing_models:
		if station_name in model:
			print(f"Model found for station {station_name} at {model}")
			break


## now find a model based on station name
station_names = "01031300"

check_existing_models(station_names)

