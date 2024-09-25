import os
import shutil
VPUIDs = os.listdir("E:/MyDataBase/SWATplus_by_VPUID")
for VPUID in VPUIDs:
    NAMES = os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")
    for NAME in NAMES:
        ## read log.txt
        logfile = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/historical_performance_scores.txt"
        if os.path.exists(logfile):
            ## remove log.txt
            os.remove(logfile)
            ##shutil.rmtree(logfile)