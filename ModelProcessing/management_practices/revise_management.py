import os

def revise_management(NAME = "04121944", LEVEL = "huc12", VPUID = "0000"):
    ## copy management.sch to TxtInOut
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/landuse.lum /data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/landuse.lum")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/management.sch /data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/management.sch")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/plant.ini /data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/plant.ini")
    os.system(f"cp /home/rafieiva/MyDataBase/codebase/ModelProcessing/example_inputs/plants.plt /data/MyDataBase/SWATplus_by_VPUID/{VPUID}/{LEVEL}/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut/plants.plt")
    

if __name__ == "__main__":
    NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")
    VPUID = "0000"
    LEVEL = "huc12"
    for NAME in NAMES:
        revise_management(NAME, LEVEL, VPUID)