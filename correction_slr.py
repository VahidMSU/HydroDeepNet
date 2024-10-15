

import os
NAME = "04096405"
NAMES = "/data/MyDataBase/SWATplus_by_VPUID/0000/huc12"
NAMES = os.listdir(NAMES)
NAMES.remove("log.txt")

for NAME in NAMES:
    if NAME not in ["04106000", "04114498"]:
        continue

    source_path = f"/data2/MyDataBase/SWATGenXAppData/SWAT_input/huc12/{NAME}/PRISM"
    if os.path.exists(source_path) is False:
        print(f"Path {source_path} does not exist")
        continue
    source_files = os.listdir(source_path)
    source_files = [f for f in source_files if f.endswith(".slr")]
    source_files_path = [os.path.join(source_path, f) for f in source_files]

    target_1_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Default/TxtInOut"

    ## copy source files to target_1_path
    for source_file in source_files_path:
        os.system(f"cp {source_file} {target_1_path}")