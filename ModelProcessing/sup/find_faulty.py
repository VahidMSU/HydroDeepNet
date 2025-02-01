import os 
import sys
import pandas as pd
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
NAMES.remove("log.txt")

for NAME in NAMES:
    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/"

    files = os.listdir(path)

    files = [f for f in files if f.endswith('.slr')]

    for f in files:
        df = pd.read_csv(path+f, sep="\t", skiprows=9, names = ['yr','day','ghi'])
        df = df[df.ghi >200]
        if len(df) > 0:
            print(NAME)
            break