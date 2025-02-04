import h5py 
import os
import time

NAME = "04136000"
ver = 0
path = f"/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/verification_stage_{ver}/SWATplus_output.h5"
var_name = "perc"

with h5py.File(path, "r") as f:
    for year in range(2000, 2020):
        for month in range(1, 13):
            group = f"hru_wb_30m/{year}/{month}"
            data = f[group][var_name][:]
            print(f"Year: {year}, Month: {month}, Min: {data.min()}, Max: {data.max()}")
