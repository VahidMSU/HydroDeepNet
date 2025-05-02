import os 
import sys
import pandas as pd

try:
    from app.utils import find_VPUID
except:
    from utils import find_VPUID
import pandas as pd
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths 

USER_PATH = "/data/SWATGenXApp/Users/"


# Model check functions (from model_checks.py)
def check_model_completion(username, site_no, MODEL_NAME="SWAT_MODEL_Web_Application", LEVEL="huc12"):
    """Check if a SWAT model execution completed successfully."""
    VPUID = find_VPUID(site_no)
    path = f"{USER_PATH}/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/{MODEL_NAME}/Scenarios/Default/TxtInOut/simulation.out"
    
    if not os.path.exists(path):
        return False, "Model execution did not complete successfully"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Execution successfully completed" in line:
                return True, "Model execution completed successfully"
    return False,"Model execution did not complete successfully"

def check_qswat_model_files(username, site_no, MODEL_NAME="SWAT_MODEL_Web_Application", LEVEL="huc12"):
    """Check if QSWAT+ processing completed successfully by verifying required files."""
    VPUID = find_VPUID(site_no)
    path = f"{USER_PATH}/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/{MODEL_NAME}/Watershed/Shapes/"
    required_files = ['subs1.shp', 'rivs1.shp', 'hrus1.shp', 'hrus2.shp', 'lsus1.shp', 'lsus2.shp']
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            return False, f"QSWAT+ processing did not complete successfully. File {file} does not exist" 
        
    return True, "QSWAT+ processing completed successfully"

def check_meterological_data(username, site_no, MODEL_NAME="SWAT_MODEL_Web_Application", LEVEL="huc12"):
    """Check if all required meteorological data files exist."""
    VPUID = find_VPUID(site_no)
    path = f"{USER_PATH}/{username}/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{site_no}/PRISM/"
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
                    pass
                else:
                    missing_files.append(line)

    if len(missing_files) == 0:
        return True, "All required meteorological data files exist"
    else:
        return False, f"Missing meteorological data files: {missing_files}"

# MODFLOW coverage check (from check_MODFLOW_coverage.py)
def MODFLOW_coverage(station_no):
    """Check if MODFLOW data is available for the given station (currently limited to Michigan LP)."""
 

    CONUS_streamflow_data = pd.read_csv(SWATGenXPaths.USGS_CONUS_stations_path, dtype={'site_no': str,'huc_cd': str})
    lat = CONUS_streamflow_data.loc[CONUS_streamflow_data['site_no'] == station_no, 'dec_lat_va'].values[0]
    lon = CONUS_streamflow_data.loc[CONUS_streamflow_data['site_no'] == station_no, 'dec_long_va'].values[0]    

    # Check whether it is in Michigan LP
    if (lat > 41.696118 and lat < 47.459853 and lon > -90.418701 and lon < -82.122818):
        return True
    else:
        return False

# Additional utility functions for model verification
def verify_model_outputs(username, site_no, MODEL_NAME="SWAT_MODEL_Web_Application", LEVEL="huc12"):
    """
    Comprehensive verification of model outputs.
    Returns a dictionary with verification results.
    """
    # Run all verification checks
    model_execution, exec_message = check_model_completion(username, site_no, MODEL_NAME, LEVEL)
    qswat_files, qswat_message = check_qswat_model_files(username, site_no, MODEL_NAME, LEVEL)
    met_data, met_message = check_meterological_data(username, site_no, MODEL_NAME, LEVEL)
    modflow_available = MODFLOW_coverage(site_no)
    
    # Combine results
    return {
        "model_execution": {
            "success": model_execution,
            "message": exec_message
        },
        "qswat_processing": {
            "success": qswat_files,
            "message": qswat_message
        },
        "meteorological_data": {
            "success": met_data,
            "message": met_message
        },
        "modflow_available": modflow_available,
        "overall_success": all([model_execution, qswat_files, met_data]),
        "missing_components": [
            component for component, status in {
                "Model Execution": not model_execution,
                "QSWAT Processing": not qswat_files,
                "Meteorological Data": not met_data
            }.items() if status
        ]
    }