import os 
import shutil

NAMES = os.listdir('/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/')
NAMES.remove('log.txt')

for name in NAMES:
    for ver in ['1', '2', '3']:
        files = f'/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/recharg_output_SWAT_gwflow_MODEL/verification_stage_{ver}'
        if os.path.exists(files):
            shutil.rmtree(files)
        files = f'/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{name}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_{ver}'
        if os.path.exists(files):
            shutil.rmtree(files)



