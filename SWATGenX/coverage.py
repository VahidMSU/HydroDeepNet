from SWATGenX.utils import get_all_VPUIDs
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
import os   
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
def plot_all_stations(all_stations, prism_shape_path, num_models):
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
    today = pd.Timestamp("today").strftime("%Y-%m-%d")
    plt.savefig(f'/data/SWATGenXApp/codes/SWATGenX/Extracted_SWAT_models_{today}.png')
    plt.close()


from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
def check_simulation_output(VPUID, LEVEL, NAME, MODEL_NAME):
    Paths = SWATGenXPaths()
    #print(f"Checking simulation output for {NAME}")
    """Checks the simulation output for successful execution."""
    execution_checkout_path = Paths.construct_path(
        Paths.swatgenx_outlet_path,
        VPUID,
        LEVEL,
        NAME,
        MODEL_NAME,
        "Scenarios",
        "Default",
        "TxtInOut",
        "simulation.out",
    )
    sim_file_exists = os.path.exists(execution_checkout_path)
    state = False
    failed_models = 0   
    if sim_file_exists:
        with open(execution_checkout_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Execution successfully completed" in line:
                    print(f"Model already exists and successfully executed for {NAME}")
                    state = True
                    

    if sim_file_exists and not state:
        print(f"Model already exists but did not execute successfully for {NAME}")
        #os.system(f"rm -r {SWATGenXPaths.construct_path(SWATGenXPaths.swatgenx_outlet_path, VPUID, LEVEL, NAME)}")

    if not state:
        print(f"Model does not exist for {NAME}")
        #os.system(f"rm -r {SWATGenXPaths.construct_path(SWATGenXPaths.swatgenx_outlet_path, VPUID, LEVEL, NAME)}")
    
    return state

def generate_coverage_map():
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
            state = check_simulation_output(VPUID, "huc12", NAME, "SWAT_MODEL")
            if not state:
                print(f"{NAME} failed")
                continue
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


    plot_all_stations(all_stations, prism_shape_path, num_models)

generate_coverage_map()







