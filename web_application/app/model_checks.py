import os 
try:
    from utils import find_VPUID
except:
    from app.utils import find_VPUID

def check_model_completion(username,  site_no, MODEL_NAME = "SWAT_MODEL_Web_Application", LEVEL="huc12"):

    VPUID = find_VPUID(site_no)
    path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/{MODEL_NAME}/Scenarios/Default/TxtInOut/simulation.out"
    
    if not os.path.exists(path):
        return False, "Model execution did not complete successfully"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Execution successfully completed" in line:
                return True, "Model execution completed successfully"
    return False,"Model execution did not complete successfully"

def check_qswat_model_files(username,site_no, MODEL_NAME = "SWAT_MODEL_Web_Application", LEVEL="huc12"):
    
    VPUID = find_VPUID(site_no)
    path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/{MODEL_NAME}/Watershed/Shapes/"
    required_files = ['subs1.shp', 'rivs1.shp', 'hrus1.shp', 'hrus2.shp', 'lsus1.shp', 'lsus2.shp']
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            print(f"File {file} does not exist")
            return False, f"QSWAT+ processing did not complete successfully. File {file} does not exist" 
        
    return True, "QSWAT+ processing completed successfully"

def check_meterological_data(username, site_no, MODEL_NAME="SWAT_MODEL_Web_Application", LEVEL="huc12"):
    
    VPUID = find_VPUID(site_no)
    path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/PRISM/"
    cli_files = os.listdir(path)
    cli_files = [x for x in cli_files if x.endswith(".cli")]
    missing_files = []
    for cli_file in cli_files:
        with open(os.path.join(path, cli_file), "r") as f:
            lines = f.readlines()[2:]
            ## remove \n
            lines = [x.strip() for x in lines]  
            for line in lines:
                if os.path.exists(os.path.join(path, line)):
                    print(f"File {line} exists")
                else:
                    missing_files.append(line)
                    print(f"File {line} does not exist")

    if len(missing_files) == 0:
        return True, "All required meteorological data files exist"
    else:
        return False, f"Missing meteorological data files: {missing_files}"
    
if __name__ == "__main__":

    username = "vahidr32"
    site_no = "05536265"
    MODEL_NAME = "SWAT_MODEL_Web_Application"

    swat_model_exe_flag, message = check_model_completion(username, site_no, MODEL_NAME)
    qswat_plus_outputs_flag, message = check_qswat_model_files(username, site_no, MODEL_NAME)
    meterological_data_flag, message = check_meterological_data(username, site_no, MODEL_NAME)
