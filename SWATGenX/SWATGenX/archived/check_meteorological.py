

import os 

path = "/data/SWATGenXApp/Users/admin/SWATplus_by_VPUID/0408/huc12/04141000/PRISM/"

cli_files = [x for x in os.listdir(path) if x.endswith(".cli")]

for cli_file in cli_files:
    ### skip the first 2 lines
    with open(os.path.join(path, cli_file), 'r') as file:
        lines = file.readlines()
        for line in lines[2:]:
            file = line.split()
            ## check if the file exists in the path
            file_path = os.path.join(path, file[0])
            if not os.path.exists(file_path):
                print(f"File {file} does not exist in the path {path}")