import shutil
import os

NAMES = os.listdir(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/0000/huc12/')
NAMES.remove('log.txt')

for NAME in NAMES:
    ## if SWAT_gwflow_MODEL_30m folder exists
    if os.path.exists(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL_30m'):
        #shutil.rmtree(f'/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL_30m')
        print(f"{NAME} Removed")