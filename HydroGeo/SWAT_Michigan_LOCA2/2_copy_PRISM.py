import shutil
import os


NAMES = os.listdir("/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12")
NAMES.remove('log.txt')
for NAME in NAMES:
    VPUID = f"0{NAME[:3]}"
    PRISM_path_source =  f"/data/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/PRISM"
    PRISM_path_target =  f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/PRISM"

    # copy the PRISM folder from the source to the target
    ## if the target folder exists, delete it
    if os.path.exists(PRISM_path_target):
        shutil.rmtree(PRISM_path_target)

    shutil.copytree(PRISM_path_source, PRISM_path_target)