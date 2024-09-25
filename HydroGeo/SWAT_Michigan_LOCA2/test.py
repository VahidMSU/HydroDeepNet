cc_model_path_new = fr"E:/MyDataBase/SWATplus_by_VPUID/0405/huc12/40500010102/climate_change_models/CanESM5_ssp585_r5i1p1f1"
import os
import shutil
import subprocess
subprocess.run(["swatplus.exe"], cwd=cc_model_path_new)
print("All done")
import time
time.sleep(5)