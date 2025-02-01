import os
VPUIDs = os.listdir("/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/")
for VPUID in VPUIDs:
    LEVELS = os.listdir(f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/{VPUID}")
    for LEVEL in LEVELS:
        NAMES = os.listdir(f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/")
        for NAME in NAMES:
            MODEL_NAMES = ['SWAT_MODEL', 'SWAT_MODEL_Web_Application']
            for MODEL_NAME in MODEL_NAMES:
                path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/{MODEL_NAME}"
                if os.path.exists(path):
                    print(f"SWAT+ The directory removed: {path}")
                    os.system(f"rm -r {path}")
        