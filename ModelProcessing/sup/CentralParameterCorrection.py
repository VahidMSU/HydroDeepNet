import os 
import time 
import pandas as pd

def cal_parameters(base_path):
    path = f"{base_path}/cal_parms_SWAT_gwflow_MODEL.cal"
    df_cal = pd.read_csv(path,  skiprows=1, sep='\s+')
    number_of_cal_parameters = len(df_cal.name)
    actual_column_names = df_cal['name'].values 
    return number_of_cal_parameters, actual_column_names


def correct_performance(base_path):
    path = f"{base_path}/CentralPerformance.txt"  
    try:
        CentralPerformance = pd.read_csv(path, sep='\t')
        unique = CentralPerformance['station'].unique()
    except:
        CentralPerformance = pd.read_csv(path, skiprows=1, delim_whitespace=True)
        unique = CentralPerformance['station'].unique()
    if "filled" in unique:

        CentralPerformance = CentralPerformance[CentralPerformance['station'].str.contains("_filled") | CentralPerformance['station'].str.contains("Yield") | CentralPerformance['station'].str.contains("ET")]
        CentralPerformance['station'] = CentralPerformance['station'].str.replace("_filled", "", regex=False)

        with open(path, 'w') as f:
            CentralPerformance.to_csv(f, sep='\t', index=False)
        keys = CentralPerformance['key'].unique()
        return keys
    else:
        print(f"#############names associated with filled are not present in {path}###############")
    return None



def correct_parameters(base_path, number_of_cal_parameters, actual_column_names, keys):
    CentralParameter = f"{base_path}/CentralParameters.txt"
    with open(CentralParameter, 'r') as f:
        lines = f.readlines()
        columns = actual_column_names   
        dict_parameters = {}
        for line in lines:
            numbers_of_columns = len(line.split())
            if numbers_of_columns != number_of_cal_parameters + 1:
                continue    
            if any(key in line for key in keys):
                key = line.split()[0]
                parameters = line.split()[1:]
                dict_parameters[key] = {columns[i]: parameters[i] for i in range(len(columns))}


    # Convert the dictionary to a DataFrame
    df2 = pd.DataFrame.from_dict(dict_parameters, orient='index')
    df2.index.name = 'key'
    with open(f"{base_path}/CentralParameters_corrected.txt", 'w') as f:
        f.write(f"key\t")
        for column in actual_column_names:
            f.write(f"{column}\t")
        df2.to_csv(f, sep='\t', index=True, columns=actual_column_names)
        print(f"Corrected CentralParameters.txt written to {base_path}/CentralParameters_corrected.txt")

def rename_remove(base_path):
    os.makedirs(f"{base_path}/archive", exist_ok=True)
    os.rename(f"{base_path}/CentralPerformance.txt", f"{base_path}/archive/CentralPerformance_old.txt")
    os.rename(f"{base_path}/CentralParameters.txt", f"{base_path}/archive/CentralParameters_old.txt")
    os.rename(f"{base_path}/CentralPerformance_corrected.txt", f"{base_path}/CentralPerformance.txt")
    os.rename(f"{base_path}/CentralParameters_corrected.txt", f"{base_path}/CentralParameters.txt")


def change_column_name(path):
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return False
    CentralPerformance = pd.read_csv(path, sep='\t')
    if "KEG" in CentralPerformance.columns:
        CentralPerformance.rename(columns={'KEG':'KGE'}, inplace=True)
        with open(path, 'w') as f:
            CentralPerformance.to_csv(f, sep='\t', index=False)
        print(f"Column name KEG changed to KGE in {path}/CentralPerformance.txt")
    



def check_all_components(base_path):
    CentralPerformance = pd.read_csv(os.path.join(base_path , "CentralPerformance.txt"), delim_whitespace=True)
  
    # Group by key
    CentralPerformance = CentralPerformance.groupby('key')
    
    # Remove keys that only have ET and Yield
    keys = list(CentralPerformance.groups.keys())
    filtered_keys = []
    for key in keys:
        df = CentralPerformance.get_group(key)
        if len(df) < 3:
            print(f"key {key} has less than 3 components, removing.")
        else:
            filtered_keys.append(key)
    
    # Filter the CentralPerformance DataFrame
    CentralPerformance_filtered = pd.concat([CentralPerformance.get_group(key) for key in filtered_keys])
    
    # Rewrite corrected version
    print(f"writing corrected version")
    with open(base_path + "CentralPerformance.txt", 'w') as f:
        CentralPerformance_filtered.to_csv(f, sep='\t', index=False)
    
    return True

def remove_first_line(path):
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return False
    with open(path, 'r') as f:
        lines = f.readlines()
        if "corrected version" in lines[0]:
            print("Already corrected")
            ## remove the first line and write again
            with open(path, 'w') as f:
                f.writelines(lines[1:])
            return True
    return False

import os


if __name__ == "__main__":
    NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
    NAMES.remove("log.txt")
    for NAME in NAMES:


        
        base_path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/"
        remove_first_line(f"{base_path}/CentralPerformance.txt")
        change_column_name(f"{base_path}/CentralPerformance.txt")
