import shutil
import os

NAMES = os.listdir(fr'/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/')
NAMES.remove('log.txt')

for NAME in NAMES:
    ## if SWAT_gwflow_MODEL_30m folder exists
    if os.path.exists(f'/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_gwflow_MODEL_30m'):
        #shutil.rmtree(f'/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/SWAT_gwflow_MODEL_30m')
        print(f"{NAME} Removed")