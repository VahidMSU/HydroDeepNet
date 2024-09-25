import os
import pandas as pd

def check_historical(VPUID, NAME, stage="historical"):
    path = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/{stage}_performance_scores.txt"
    if not os.path.exists(path):
        print(f"VPUID: {VPUID}, NAME: {NAME} does not have a performance_scores.txt file.")
        return False
    else:
        df = pd.read_csv(path, sep="\t", skiprows=1)
        num_stations = df['station'].nunique()
        if len(df)/num_stations >180:
            print(f"Number of unique stations: {num_stations}", f"Number of rows: {df.shape[0]}")
            return True
        else:
           # os.remove(path)
            print(f"VPUID: {VPUID}, NAME: {NAME} has less than 180 rows. Deleting file.")
            return False


def read_historical():
    VPUIDs = os.listdir("E:/MyDataBase/SWATplus_by_VPUID")
    print(VPUIDs)
    for VPUID in VPUIDs:
        NAMES =  os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
        NAMES.remove("log.txt")
        for NAME in NAMES:
            if check_historical(VPUID, NAME):
                print(f"VPUID: {VPUID}, NAME: {NAME} has a historical simulation.")
                continue





read_historical()