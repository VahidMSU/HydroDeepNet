import h5py
import pandas as pd
SWAT_h5_path = "E:\MyDataBase\climate_change\LOCA2_MLP.h5"
cc_model_list    = "E:/MyDataBase/climate_change/list_of_all_models.txt"
def get_model_scenario_ensemble(cc_model_list):
    data = []
    with open(cc_model_list, 'r') as file:
        for line in file:
            if parts := line.split():
                model = parts[1]
                scen = parts[2]
                ensembles = parts[3:]
                data.extend([model, scen, ens] for ens in ensembles)
                if "99" in parts[0]:
                    break
    return data


cc_models = get_model_scenario_ensemble(cc_model_list)

for model, scenario, ensembles in cc_models:

    for ensemble in ensembles.split(","):
        #print(f"Processing: model={model}, scenario={scenario}, ensemble={ensemble}")
        with h5py.File(SWAT_h5_path, 'r') as f:
            if scenario != "historical":
                try:
                    keys = f['e_n_cent'][model][scenario][ensemble]['daily']['2015_2044'].keys()
                except Exception as e:
                    print(f"Error: {e}, model={model}, scenario={scenario}, ensemble={ensemble}")
                    continue

                if "tasmin" not in keys or "tasmax" not in keys or "pr" not in keys:
                    print(f"Missing variable for model={model}, scenario={scenario}, ensemble={ensemble}", keys)
                    continue
