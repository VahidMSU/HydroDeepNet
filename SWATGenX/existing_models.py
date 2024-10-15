path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/"
import os
VPUIDs = os.listdir(path)
existing_models = []
for VPUID in VPUIDs:
	# now find model inside huc12 directory
	huc12_path = os.path.join(path, VPUID, "huc12")
	models = os.listdir(huc12_path)
	existing_models.extend(os.path.join(huc12_path, model) for model in models)

print(existing_models)

## now find a model based on station name
station_names = ["04164300"]

for station_name in station_names:
	for model in existing_models:
		if station_name in model:
			print(f"Model found for station {station_name} at {model}")
			break
