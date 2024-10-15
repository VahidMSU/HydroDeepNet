import os
import shutil

TARGET_PATH = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/"
VPUIDs = os.listdir(TARGET_PATH)
existing_models = []
SWAT_MODEL_NAME = "SWAT_MODEL_30m"
for VPUID in VPUIDs:
    NAMES = os.listdir(os.path.join(TARGET_PATH, VPUID, "huc12"))
    for NAME in NAMES:
        if not os.path.exists(os.path.join(TARGET_PATH, VPUID, "huc12", NAME, SWAT_MODEL_NAME)):
            #print(f"Model does not exist for {NAME}")
            continue

        if not os.path.exists(os.path.join("/data/SWATGenXApp/GenXAppData/SWAT_input/huc12", NAME, SWAT_MODEL_NAME)):
            shutil.copytree(os.path.join(TARGET_PATH, VPUID, "huc12", NAME, SWAT_MODEL_NAME), os.path.join("/data/SWATGenXApp/GenXAppData/SWAT_input/huc12", NAME, SWAT_MODEL_NAME))
            print(f"Model successfully copied to SWAT_input for {NAME}")
        else:
            print(f"Model already exists for {NAME} in SWAT_input")