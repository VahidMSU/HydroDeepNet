from SWATGenX.utils import get_all_VPUIDs
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import os   
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

VPUIDs = get_all_VPUIDs()
all_stations = []

prism_shape_path = SWATGenXPaths.PRISM_mesh_pickle_path
num_models = 0
for VPUID in VPUIDs:
    print(VPUID)
    base_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/"
    if not os.path.exists(base_path):
        print(f"{VPUID} does not have any huc12 data")
        continue
    NAMES = os.listdir(base_path)
    for NAME in NAMES:
        path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/streamflow_data/stations.shp"
        if not os.path.exists(path):
            print(f"{NAME} does not have streamflow data")
            continue
        num_models += 1
        station_shp = gpd.read_file(path)
        all_stations.append(station_shp)
        if os.path.exists(path):
            print(f"{NAME} has streamflow data")
        else:
            print(f"{NAME} does not have streamflow data")

all_stations = gpd.GeoDataFrame(pd.concat(all_stations))
conuse_shape = pd.read_pickle(prism_shape_path)
## get the union of conuse shaoe
conuse_shape = conuse_shape.dissolve()
conuse_shape = conuse_shape.reset_index()
fig, ax = plt.subplots(figsize=(10, 10))
conuse_shape.plot(ax=ax, color='white', edgecolor='black')
all_stations.plot(ax=ax, color='blue')
plt.title(f"Extracted {num_models} SWAT+ models containing {len(all_stations)} stations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid()  
plt.tight_layout()
plt.savefig("all_stations.png", dpi=300)

