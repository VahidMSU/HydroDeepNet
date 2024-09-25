import arcpy
import os 

NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
NAMES.remove("log.txt")
for NAME in NAMES:
    generate_reference_raster(NAME)