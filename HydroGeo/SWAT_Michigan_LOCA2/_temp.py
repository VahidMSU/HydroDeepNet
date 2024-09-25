import os
import shutil

VPUIDs =  os.listdir("E:/MyDataBase/SWATplus_by_VPUID")
for VPUID in VPUIDs:
    NAMES = os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")
    for NAME in NAMES:
        cc_models_path = os.path.join(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}", 'climate_change_models')
        cc_models = os.listdir(cc_models_path)
        cc_models = [model for model in cc_models if "," in model]
        if not cc_models:
            print(f"No models found in {cc_models_path}")
            continue
        for cc_model in cc_models:
            ### remove directory if it exists
            print(f"Removing {cc_model}")
            shutil.rmtree(os.path.join(cc_models_path, cc_model))