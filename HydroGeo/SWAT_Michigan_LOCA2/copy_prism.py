import os
import shutil

VPUIDs = os.listdir("E:/MyDataBase/SWATplus_by_VPUID")
for VPUID in VPUIDs:

    NAMES = os.listdir(f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")

    for NAME in NAMES:

        ## read log.txt
        logfile = f"E:/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/PRISM/"
        destination = f"//35.9.219.75/Data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/PRISM/"

        os.makedirs(destination, exist_ok=True)

        for file in os.listdir(logfile):
            print(f"Copying {file} to {destination}")
            shutil.copy(os.path.join(logfile, file), destination)
