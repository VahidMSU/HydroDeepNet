import os
from concurrent.futures import ThreadPoolExecutor

class ClimateChangeModelCleaner:
    def __init__(self, base_path, selected_name, cc_model):
        self.base_path = base_path
        self.selected_name = selected_name
        self.cc_model = cc_model

    def clean_up(self, path):
        files = os.listdir(path)
        for file in files:
            #if not (file.endswith('.pcp') or file.endswith('.tmp')):
            os.remove(os.path.join(path, file))

    def process_cc_model(self, cc_models_path, cc_model):
        self.clean_up(os.path.join(cc_models_path, cc_model))

    def process_name(self, huc12_path, name):
        cc_models_path = os.path.join(huc12_path, name, 'climate_change_models')
        cc_models = os.listdir(cc_models_path)
        cc_models = [model for model in cc_models if self.cc_model in model]
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_cc_model, [cc_models_path] * len(cc_models), cc_models)

    def process_vpuid(self, vpuid):
        print(f"Processing {vpuid}")
        huc12_path = os.path.join(self.base_path, vpuid, 'huc12')
        names = os.listdir(huc12_path)

        if "log.txt" in names:
            names.remove("log.txt")

        filtered_names = [name for name in names if name == self.selected_name]

        print(f"Processing {filtered_names}")
        with ThreadPoolExecutor(10) as executor:
            executor.map(self.process_name, [huc12_path] * len(filtered_names), filtered_names)

    def cleanup_cc_models(self):
        vpuid_list = ['0408'] #os.listdir(self.base_path)
        with ThreadPoolExecutor(10) as executor:
            executor.map(self.process_vpuid, vpuid_list)


if __name__ == "__main__":
    base_path = "E:/MyDataBase/SWATplus_by_VPUID"
    selected_name = "40601010503"
    cc_model = "CNRM-CM6-1"

    cleaner = ClimateChangeModelCleaner(base_path, selected_name, cc_model)
    cleaner.cleanup_cc_models()
