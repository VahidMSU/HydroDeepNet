import os
import pandas as pd
import numpy as np
import geopandas as gpd
def load_climate_data(txtinout_path,df, wst, typ, start_year, end_year):
    if typ == "tmin" or typ == "tmax":
        file = df[df["wst"] == wst]["tmp"].values[0]
    else:
        file = df[df["wst"] == wst][typ].values[0]
    row = file.split("_")[0][1:]
    col = file.split("_")[1][1:].split(".")[0]  
    file_path = f"{txtinout_path}/{file}"
    if typ in ["tmin", "tmax"]:
        return load_temp_data(txtinout_path, row, col, typ, start_year, end_year)
    if os.path.exists(file_path):
        pcp = pd.read_csv(file_path, sep="\s+", skiprows=3, names=["YEAR", "DAY", "value"])
        # Filter between start_year to end_y
        pcp = pcp[(pcp["YEAR"] >= start_year) & (pcp["YEAR"] <= end_year)]
        return pcp["value"].values
    else:
        print(f"Warning: File {file_path} does not exist.")
        return [0] * 365  # Assuming 365 days of data, replace with the appropriate length

def load_climate_datas(txtinout_path, wst, start_year=2000, end_year=2001):
    wst_path = f"{txtinout_path}/weather-sta.cli"
    df = pd.read_csv(wst_path, skiprows=1, sep="\s+").rename(columns={"name":"wst"})
    dic = {"pcp": [], "tmax": [], "hmd": [], "wnd": [], "slr": [], "tmin": []}
    for typ in dic:
        dic[typ] = load_climate_data(txtinout_path,df,  wst, typ, start_year, end_year)
    ### create a 2d array
    #print(f"climate data for wst {wst} is loaded")
    #print(f"shape of the climate data is {np.array(list(dic.values())).T.shape}")
    return np.array(list(dic.values())).T


def load_temp_data(txtinout_path, row, col, typ, start_year, end_year):
    file_path = f"{txtinout_path}/r{int(row)}_c{int(col)}.tmp"
    if os.path.exists(file_path):
        pcp = pd.read_csv(file_path, sep="\s+", skiprows=3, names=["YEAR", "DAY", "tmax", "tmin"])
        pcp = pcp[(pcp["YEAR"] >= start_year) & (pcp["YEAR"] <= end_year)]
        return pcp[typ].values
    else:
        print(f"Warning: File {file_path} does not exist.")
        return []
