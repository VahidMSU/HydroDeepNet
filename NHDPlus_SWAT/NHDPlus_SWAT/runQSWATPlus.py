import subprocess
import os
def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME):

    runQSWATPlus_path = "/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/runQSWATPlus.bat"
    ## now read the last line of the batch file and insert arguments
    with open(runQSWATPlus_path, "r") as f:
        lines = f.readlines()
        lines[-1] = f"%OSGEO4W_ROOT%\\bin\\python3.exe -c \"from QSWATPlus3_9 import runHUCProject; runHUCProject(VPUID = '{VPUID}', LEVEL = '{LEVEL}', NAME = '{NAME}', MODEL_NAME = '{MODEL_NAME}')\"\n"
        print(lines[-1])
    ## now write it back to a new batch file
    runQSWATPlus_path_new = f"/data/MyDataBase/SWATGenXAppData/codes/NHDPlus_SWAT/{NAME}_runHUCProject.bat"
    with open(runQSWATPlus_path_new, "w") as f:
        f.writelines(lines)
    ## now run the batch file
    subprocess.run(runQSWATPlus_path_new)
    # now remove the batch file
    os.remove(runQSWATPlus_path_new)