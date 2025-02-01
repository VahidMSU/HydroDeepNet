import os
from SWATGenXConfigPars import SWATGenXPaths
from utils import get_all_VPUIDs


def check_simulation_output(VPUID, LEVEL, NAME, MODEL_NAME):
    #print(f"Checking simulation output for {NAME}")
    """Checks the simulation output for successful execution."""
    paths = SWATGenXPaths()
    execution_checkout_path = paths.construct_path(
        paths.swatgenx_outlet_path,
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
        os.system(f"rm -r {paths.construct_path(paths.swatgenx_outlet_path, VPUID, LEVEL, NAME)}")

    if not state:
        print(f"Model does not exist for {NAME}")
        os.system(f"rm -r {paths.construct_path(paths.swatgenx_outlet_path, VPUID, LEVEL, NAME)}")
    
    return state

if __name__ == "__main__":
    VPUIDs = get_all_VPUIDs()
    total_models = 0    
    successful_models = 0   
    for VPUID in VPUIDs:
        base_path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/"
        if not os.path.exists(base_path):
            #print(f"{VPUID} does not have any huc12 data")
            continue
        NAMES = os.listdir(base_path)
        for NAME in NAMES:
            state = check_simulation_output(VPUID, "huc12", NAME, "SWAT_MODEL")
            total_models += 1
            if state:
                successful_models += 1

    print(f"Total models: {total_models}, Successful models: {successful_models}")

