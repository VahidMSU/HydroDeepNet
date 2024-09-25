import numpy as np
import pandas as pd
from SALib.analyze import morris
from SALib.sample import morris as ms
import os 
import numpy as np
import pandas as pd
from skopt.space import Real


def read_control_file(cal_parms):
        param_files = {}
        operation_types = {}
        problem = {'num_vars': 0, 'names': [], 'bounds': []}

        for _, row in cal_parms.iterrows():
                name = row['name']
                file_name = row['file_name']
                min_val = float(row['min'])  # Explicitly cast to float
                max_val = float(row['max'])
                operation = row['operation']

                if file_name not in param_files:
                        param_files[file_name] = []
                param_files[file_name].append(name)

                operation_types[name] = operation
                problem['names'].append(name)
                problem['bounds'].append((min_val, max_val))
        problem['num_vars'] = len(problem['names'])

#        print(f"problem: {problem}  ")
        return param_files, operation_types, problem


def get_space_and_problem(cal_parms):
    param_files, operation_types, problem = read_control_file(cal_parms)

    space = [Real(low, high, name=name) for (low, high), name in zip(problem['bounds'], problem['names'])]
#    print(f"Number of parameters: {len(space)}")

    return space, problem

def test_sensitivity(path, problem):
        
        initial_points_path = os.path.join(path, "initial_points_SWAT_gwflow_MODEL.csv")
        initial_values_path = os.path.join(path, "initial_values_SWAT_gwflow_MODEL.csv")

        if not os.path.exists(initial_points_path) and not os.path.exists(initial_values_path):
            #print(f"Initial points or values not found in {path}")
            return
        
        initial_points = np.loadtxt(initial_points_path, delimiter=",")
        initial_values = np.loadtxt(initial_values_path, delimiter =',')
        
        print(os.path.basename(path), 
            f" initial_points shape: {initial_points.shape}",
            "number of values euqal to 1e6:",
            initial_values[initial_values == 1e6].shape,
        )
        
        try:
            Si = morris.analyze(problem, initial_points, initial_values, print_to_console=False)
            Si = pd.DataFrame(Si)
            #print(Si)
            return Si
        
        except Exception as e:
            print(f"Error in sensitivity analysis: {e}")
            return None

def main():
    Sis = []
    NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
    if "log.txt" in NAMES:
        NAMES.remove("log.txt")
    for NAME in NAMES:
        path = f"/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/{NAME}/"
        parameters = pd.read_csv(f"{path}/cal_parms_SWAT_gwflow_MODEL.cal", skiprows=1, sep="\s+")
        param_names = np.unique(parameters.name.values)
        space, problem = get_space_and_problem(parameters)
        Si = test_sensitivity(path, problem)
        if Si is not None:
            Si['NAME'] = NAME
            Sis.append(Si)
    if Sis:
        Sis = pd.concat(Sis)
        Sis.to_csv("Morris_Sensitivity_Analysis/sensitivity_analysis.csv")
        print("Number of unique NAME: ", len(Sis.NAME.unique()))    
        
if __name__ == "__main__":
    main()
