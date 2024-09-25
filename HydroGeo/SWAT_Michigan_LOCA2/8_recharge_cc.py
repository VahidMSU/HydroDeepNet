
import os
import h5py
import pandas as pd
cc_name = "AWI-CM-1-1-MR_ssp370_r3i1p1f1"
NAME = "40500010206"
path = f"E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/{NAME}/climate_change_models/{cc_name}"


SWAT_path = f"E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut"

SWAT_h5 = os.path.join(path, "SWAT_OUTPUT.h5")
hru_con_path = os.path.join(SWAT_path, "hru.con")  # ["id", "name",  "gis_id", "area", "lat", "lon", "elev", "hru", "wst", "cst", "ovfl", "rule", "out_tot"]
hru_con = pd.read_csv(hru_con_path, skiprows=1, delim_whitespace=True)
lat = hru_con["lat"].values
lon = hru_con["lon"].values
elev = hru_con["elev"].values
name = hru_con["name"].values
print(f" HUC con columns: {hru_con.columns}")
print(f"name: {name}")

with h5py.File(SWAT_h5, 'r') as f:
    print(f.keys())
    ### print years range
    print(f"years: {f['/hru_wb'].keys()}")  # years: <KeysViewHDF5
    precip = f['/hru_wb/mon/precip'][()]  # precipitation: (996, 15804)
    perc = f['/hru_wb/mon/perc'][()]  # percolation: (996, 15804)
    lu_mgt = f['metadata/hru/lu_mgt'][()]  # (15804,)
    soil = f['metadata/hru/soil'][()]    # (15804,)
    name = f["metadata/hru/name"][()]    # (15804,)

    ### create a dataframe with name, soil, lu_mgt, lat, lon, elev, perc
    df = pd.DataFrame({"name": name, "soil": soil, "lu_mgt": lu_mgt, "lat": lat, "lon": lon, "elev": elev})
    print(f"df: {df.shape}")
