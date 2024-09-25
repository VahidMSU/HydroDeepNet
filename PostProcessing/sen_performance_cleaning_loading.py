import pandas as pd
import geopandas as gpd
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

VPUID = '0000'
BASE_PATH = "/data/MyDataBase/SWATplus_by_VPUID"
LEVELS = ['huc12']
NAMES = os.listdir(f'{BASE_PATH}/{VPUID}/huc12/')
for NAME in NAMES:

    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/sensitivity_performance_scores.txt"
    cleaned_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/cleaned_sensitivity_performance_scores.txt"
    with open(path, 'r') as f, open(cleaned_path, 'w') as f2:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i==0:
                f2.write(line)
                continue
            if i==1:
                print(f'First line: {line}')
                f2.write(line)
                column_names = line.strip().split('\t')
                print(f'Column names: {column_names}')   #['Time_of_writing', 'station', 'time_step', 'MODEL_NAME', 'SCENARIO', 'NSE', 'MPE', 'PBIAS\n']
                continue
            else:
                mod_line = line.strip().split('\t')
                if len(mod_line) <3:
                    continue
                f2.write('\t'.join(mod_line[:8]) + '\n')

    # remove and rename
    os.remove(path)
    os.rename(cleaned_path, path)
    print(f'Finished {NAME}')

    break