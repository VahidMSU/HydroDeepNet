import os 

VPUIDS = os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
for VPUID in VPUIDS:
    NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
    NAMES.remove("log.txt")
    for NAME in NAMES:
        for i in range(5):
            base_scenario_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_{i}"
            ## check and remove txt files
            if os.path.exists(base_scenario_path):

                files = os.listdir(base_scenario_path)
                txt_files = [file for file in files if file.endswith("_test")]
                for txt_file in txt_files:
                    os.remove(os.path.join(base_scenario_path, txt_file))