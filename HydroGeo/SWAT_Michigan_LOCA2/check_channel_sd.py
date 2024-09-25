import os
import pandas as pd
def check_channel_sd(VPUID, NAME, cc_model):
    cc_model_path = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/{cc_model}/channel_sd_day.txt"
    if os.path.exists(cc_model_path):
        channel_sd = pd.read_csv(cc_model_path,sep='\s+', skiprows=1)
        channel_sd = channel_sd.iloc[1:]
        channel_sd["yr"] = channel_sd["yr"].astype(int)
        if channel_sd["yr"].min() < 2000 or channel_sd["yr"].max() > 2015:
            print(f"VPUID: {VPUID}, NAME: {NAME}, cc_model: {cc_model}")
            print(f"invalid year range: {channel_sd['yr'].min()} - {channel_sd['yr'].max()}")
            remove_flag = True
        else:
            print(f"VPUID: {VPUID}, NAME: {NAME}, cc_model: {cc_model}")
            print("channel sd is in the range of 2000 and 2015")
            remove_flag = False
    else:
        print(f"VPUID: {VPUID}, NAME: {NAME}, cc_model: {cc_model}")
        print("-----------------------------------------------------")
        remove_flag = True


VPUIDs = os.listdir("E:/MyDataBase/SWATplus_by_VPUID")
for VPUID in VPUIDs:
    NAMES = os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")
    for NAME in NAMES:
        cc_models = os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models")
        for cc_model in cc_models:
            cc_model_path = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/{cc_model}/channel_sd_day.txt"
            if os.path.exists(cc_model_path):
                ## read and find out the range of year is between 2000 and 2015
                remove_flag = check_channel_sd(VPUID, NAME, cc_model)
# Compare this snippet from SWAT_Michigan_LOCA2/chech_channel_sd.py: