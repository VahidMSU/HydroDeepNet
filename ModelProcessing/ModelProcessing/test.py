



import os 

import pandas as pd 

path = "/data/SWATGenXApp/Users/vahidr32/SWATplus_by_VPUID/0712/huc12/05536265/SWAT_MODEL_Web_Application/Scenarios/Scenario_0b9c1b1f-a7ba-4751-a839-56d0dc3564ac/channel_sd_day.txt"
#path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/04096405/SWAT_gwflow_MODEL/Scenarios/verification_stage_0/channel_sd_day.txt"

df = pd.read_csv(path, delim_whitespace=True, skiprows=[0,2])[['day', 'mon', 'yr', 'gis_id', 'flo_out']]

print(df.head())
for gis_id in df['gis_id'].unique():
    print(gis_id)
    df_gis = df[df['gis_id'] == gis_id]
    ## if all not zero
    if df_gis['flo_out'].sum() == 0:
        print(f"gis_id {gis_id} has all zero values.")
        continue