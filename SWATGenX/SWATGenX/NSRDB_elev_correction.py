import pandas as pd 
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import geopandas as gpd

def get_elev(row, col):
    import rasterio
    with rasterio.open(SWATGenXPaths.PRISM_dem_path) as src:
        elev_data = src.read(1)
    return elev_data[row, col]

def correct_elevation(file, path):
    try:
        row = file.split("r")[1].split("c")[0].split("_")[0]
        col = file.split("r")[1].split("c")[1].split(".")[0]

        with open(os.path.join(path, file), 'r') as f:
            lines = f.readlines()
            elev = lines[2].split("\t")[-1]
            act_elev = get_elev(int(row), int(col))
            print(f"elev: {elev} row: {row} col: {col} act_elev: {act_elev}")
            lines[2] = lines[2].replace(elev, str(act_elev)) + "\n"
        with open(os.path.join(path, file), 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error: {e}")

        
import os 
VPUID = "1030"
base_path = SWATGenXPaths.swatgenx_outlet_path
print(f"base_path: {base_path}")
for VPUID in os.listdir(base_path):

    path1 = f"{base_path}/{VPUID}/huc12"
    print(f"path1: {path1}")    
    for NAME in os.listdir(path1):
        path = f"{base_path}/{VPUID}/huc12/{NAME}/PRISM"
        if not os.path.exists(path):
            ## remove the NAME from the list
            if os.path.exists(f"{base_path}/{VPUID}/huc12/{NAME}"):
                #os.rmdir(f"{base_path}/{VPUID}/huc12/{NAME}")
                os.system(f"rm -rf {base_path}/{VPUID}/huc12/{NAME}")
            continue
    
        files = os.listdir(path)
        pcps = [f for f in files if f.endswith(".pcp")]
        slrs = [f for f in files if f.endswith(".slr")] 
        wnds = [f for f in files if f.endswith(".wnd")]
        hmds = [f for f in files if f.endswith(".hmd")]

        for slr in slrs:
            correct_elevation(slr, path)

        for hmd in hmds:
            correct_elevation(hmd, path)

        for wnd in wnds:

            correct_elevation(wnd, path)


        path = f"{base_path}/{VPUID}/huc12/{NAME}/SWAT_MODEL/Scenarios/Default/TxtInOut"

        if not os.path.exists(path):
            continue

        if len(os.listdir(path)) == 0:
            continue

        
        for slr in slrs:
            correct_elevation(slr, path)

        for hmd in hmds:
            correct_elevation(hmd, path)

        for wnd in wnds:
            
            correct_elevation(wnd, path)
