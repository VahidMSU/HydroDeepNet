import os
from ModelProcessing.SWATGenXConfigPars import SWATGenXPaths
def revise_management(NAME = "04121944", LEVEL = "huc12", VPUID = "0000"):
    ## copy management.sch to TxtInOut
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/landuse.lum /{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/landuse.lum")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/management.sch /{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/management.sch")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/plant.ini /{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/plant.ini")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/plants.plt /{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/plants.plt")
    

if __name__ == "__main__":
    VPUID = "0000"  
    NAMES = os.listdir(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12")
    NAMES.remove("log.txt")
    LEVEL = "huc12"
    for NAME in NAMES:
        revise_management(NAME, LEVEL, VPUID)