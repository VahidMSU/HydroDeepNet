import os
from concurrent.futures import ThreadPoolExecutor


def process_name(NAME):
    base_path = os.path.join(path, NAME, "SWAT_gwflow_MODEL")
    
    scenarios = os.listdir(os.path.join(base_path, "Scenarios"))
    # Clear the scenarios except the Default
    for scenario in scenarios:
        if scenario != "Default":
            scenario_path = os.path.join(base_path, "Scenarios", scenario)
            os.system(f"rm -rf {scenario_path}")

    name_path = os.path.join(path, NAME)
    # Remove everything except SWAT_gwflow_MODEL, streamflow_data, MODIS_ET
    for file in os.listdir(name_path):
        if file not in ["SWAT_gwflow_MODEL", "streamflow_data", "MODIS_ET", "CentralParameters.txt", "CentralPerformance.txt"]:
            file_path = os.path.join(name_path, file)
            os.system(f"rm -rf {file_path}")

    ### clean up MODIS_ET and only keep the MODIS_ET/MODIS_ET.csv
    modis_et_path = os.path.join(name_path, "MODIS_ET")
    for file in os.listdir(modis_et_path):
        if file != "MODIS_ET.csv":
            file_path = os.path.join(modis_et_path, file)
            os.system(f"rm -rf {file_path}")

    remove_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/recharg_output"
    if os.path.exists(remove_path):
        os.system(f"rm -rf {remove_path}")


if __name__ == "__main__":
    path = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/"
    NAMES = os.listdir(path)
    NAMES.remove("log.txt")

    # Use ThreadPoolExecutor to parallelize the processing
    ### be mindful! This will remove all the previous runs and only keep the Default scenario ###
   # with ThreadPoolExecutor() as executor: ###
       # executor.map(process_name, NAMES)  ##