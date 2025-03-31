import os 
import shutil 
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12")
NAMES.remove("log.txt")

for NAME in NAMES:

    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/watershed_static_plots"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")

    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/verifications_videos"

    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")

    path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/figures_SWAT_gwflow_MODEL/{NAME}/"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed {path}")