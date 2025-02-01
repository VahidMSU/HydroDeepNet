import os 
import shutil 
from SWATGenX.SWATGenXConfigPars import SWATGenXPaths
VPUID = "0000"
NAMES = os.listdir(f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12")
NAMES.remove("log.txt")

for NAME in NAMES:
    path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL/watershed_static_plots"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")

    path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL/verifications_videos"

    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")

    path = f"{SWATGenXPaths.swatgenx_outlet_path}/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL/{NAME}/"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")