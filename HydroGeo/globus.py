import os 

for i in range(1980,2021):
    path = f"/data/CONUS404/wrfconus404/WY{i}/wrfxtrm/"
    
    os.makedirs(path, exist_ok=True)