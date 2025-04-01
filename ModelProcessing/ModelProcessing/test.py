



import os 

import pandas as pd 

#path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0712/huc12/05536265/SWAT_MODEL_Web_Application/Scenarios/Default/TxtInOut/channel_sd_day.txt"
#path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/04096405/SWAT_gwflow_MODEL/Scenarios/verification_stage_0/channel_sd_day.txt"
#path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0712/huc12/05536265/SWAT_MODEL/Scenarios/Default/TxtInOut/channel_sd_day.txt"
path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0712/huc12/05536265/SWAT_MODEL/Scenarios/Default/TxtInOut/channel_sd_day.txt"

exe = "/data/SWATGenXApp/codes/bin/swatplus"
import shutil
## copy exe to path base dir
path_base = os.path.dirname(path)
shutil.copy(exe, path_base)
import subprocess

## run exe
subprocess.run([exe, path], cwd=path_base)


df = pd.read_csv(path, delim_whitespace=True, skiprows=[0,2])[['day', 'mon', 'yr', 'gis_id', 'flo_out']]

print(df.head())
for gis_id in df['gis_id'].unique():
   # print(gis_id)
    df_gis = df[df['gis_id'] == gis_id]
    ## if all not zero
    if df_gis['flo_out'].sum() == 0:
        #print(f"gis_id {gis_id} has all zero values.")
        continue
    else:
        print(f"gis_id {gis_id} has non-zero values.")
    