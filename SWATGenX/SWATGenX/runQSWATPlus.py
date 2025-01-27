import subprocess
import os
try:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
except ImportError:
    from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
def runQSWATPlus(VPUID, LEVEL, NAME, MODEL_NAME):
    
    print(f"Running QSWATPlus for {NAME}")
    runQSWATPlus_path = SWATGenXPaths.runQSWATPlus_path
    assert os.path.exists(runQSWATPlus_path), f"File {runQSWATPlus_path} does not exist"
    ## now read the last line of the batch file and insert arguments
    with open(runQSWATPlus_path, "r") as f:
        lines = f.readlines()
        lines[-1] = f"xvfb-run -a python3 -c \"from QSWATPlus3_64 import runHUCProject; runHUCProject(VPUID = '{VPUID}', LEVEL = '{LEVEL}', NAME = '{NAME}', MODEL_NAME = '{MODEL_NAME}')\"\n"
        print(lines[-1])

    ## now write it back to a new batch file
    runQSWATPlus_path_new = f"{SWATGenXPaths.codes_path}/{NAME}_runHUCProject.sh"
    with open(runQSWATPlus_path_new, "w") as f:
        f.writelines(lines)
    

    ### make sure the chmod is 777
    os.chmod(runQSWATPlus_path_new, 0o775)
    #os.chmod("/data/SWATGenXApp/script.sh", 0o775)
    ## now run the batch file
    subprocess.run(runQSWATPlus_path_new)
    # now remove the batch file
    os.remove(runQSWATPlus_path_new)


if __name__ == "__main__":
    runQSWATPlus("0407", "huc12", "04128990", "SWAT_MODEL")
