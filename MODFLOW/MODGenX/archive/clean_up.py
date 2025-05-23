### remove previous models
import os
import shutil

def cleanup_models(MODEL_NAMES, NAMES, BASE_PATH):
	for NAME in NAMES:
		for MODEL_NAME in MODEL_NAMES:
			path_to_model = os.path.join(f'/data/SWATGenXApp/GenXAppData/{username}/', f"SWATplus_by_VPUID/0000/huc12/{NAME}/{MODEL_NAME}")
			if os.path.exists(path_to_model):
				shutil.rmtree(path_to_model)
				print(f"{MODEL_NAME}_{NAME} Removed")
			else:
				print(f"{MODEL_NAME}_{NAME} not found")

if __name__ == "__main__":
	RESOLUTION = 250
	MODEL_NAMES = [f'MODFLOW_EBK_{RESOLUTION}m', f'MODFLOW_ML_{RESOLUTION}m', f'Michigan_EBK_{RESOLUTION}m']
	NAMES = os.listdir('/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/0000/huc12/')
	NAMES.remove('log.txt')
	BASE_PATH = "/data/SWATGenXApp/GenXAppData/"
	cleanup_models(MODEL_NAMES, NAMES, BASE_PATH)
