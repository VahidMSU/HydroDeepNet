<<<<<<< HEAD
import os
import shutil
#### removing the models 


def remove_incomplete():
    VPUIDs =  os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
    for VPUID in VPUIDs:
        if VPUID !="0405":
            continue
        NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
        NAMES.remove("log.txt")
        for NAME in NAMES:
            verification_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/"
            base_scenario_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_0"

            files = os.listdir(base_scenario_path)
            ## get files ending with pcp and tmp
            base_pcp_files = [file for file in files if file.endswith("pcp")]
            base_tmp_files = [file for file in files if file.endswith("tmp")]

            cc_models = os.listdir(verification_path)
            for cc_model in cc_models:
                path = os.path.join(verification_path, cc_model)
                files = os.listdir(path)
                cc_pcp_files = [file for file in files if file.endswith("pcp")] 
                cc_tmp_files = [file for file in files if file.endswith("tmp")]
                if len(cc_pcp_files) != len(base_pcp_files) or len(cc_tmp_files) != len(base_tmp_files):
                    print(f"Deleting {cc_model}")
                    shutil.rmtree(path) 

if __name__ == "__main__":
    remove_incomplete()
=======
import os
import shutil
#### removing the models 


def remove_incomplete():
    model_num = 0
    VPUIDs =  os.listdir("/data/MyDataBase/SWATplus_by_VPUID")
    for VPUID in VPUIDs:

        NAMES = os.listdir(f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12")
        NAMES.remove("log.txt")
        for NAME in NAMES:
            verification_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/climate_change_models/"
            base_scenario_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/SWAT_gwflow_MODEL/Scenarios/Scenario_verification_stage_0"
            fig_path = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/figures_SWAT_gwflow_MODEL_verification_daily"
            if not os.path.exists(fig_path):
                print(f"#########################Deleting {fig_path}#########################")
                delete_model = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/"
                shutil.rmtree(delete_model)
                continue
            figs = os.listdir(fig_path)
            if len(figs) == 0:
                print(f"#########################Deleting {fig_path}#########################")
                delete_model = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/"
                shutil.rmtree(delete_model)
                continue
            if not os.path.exists(base_scenario_path):
                print(f"#########################Deleting {verification_path}#########################")
                delete_model = f"/data/MyDataBase/SWATplus_by_VPUID/{VPUID}/huc12/{NAME}/"
                shutil.rmtree(delete_model)
                continue

            model_num = model_num + 1
            files = os.listdir(base_scenario_path)
            ## get files ending with pcp and tmp
            base_pcp_files = [file for file in files if file.endswith("pcp")]
            base_tmp_files = [file for file in files if file.endswith("tmp")]

            cc_models = os.listdir(verification_path)
            for cc_model in cc_models:
                path = os.path.join(verification_path, cc_model)
                files = os.listdir(path)
                cc_pcp_files = [file for file in files if file.endswith("pcp")] 
                cc_tmp_files = [file for file in files if file.endswith("tmp")]
                if len(cc_pcp_files) != len(base_pcp_files) or len(cc_tmp_files) != len(base_tmp_files):
                    print(f"Deleting {cc_model}")
                    shutil.rmtree(path) 
    print(f"Total number of models {model_num}")

if __name__ == "__main__":
    remove_incomplete()
>>>>>>> 4151fd4 (initiate linux version)
