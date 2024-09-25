<<<<<<< HEAD
import h5py

linux = False

if linux:
    h5_path = "/data/HydroMetData/LOCA/LOCA2_extracted.h5"
else:
    h5_path = "E:/MyDataBase/climate_change/LOCA2_MLP_2.h5"

# Open the file in read/write mode
with h5py.File(h5_path, 'r+') as f:
    # Get the unit from the group attribute
    unit = f['e_n_cent/ACCESS-CM2/historical/r2i1p1f1/daily/1950_2014/tasmax']
=======
import h5py


h5_path = "/data/MyDataBase/LOCA2_MLP.h5"

# Open the file in read/write mode
with h5py.File(h5_path, 'r+') as f:
    # Get the unit from the group attribute
    unit = f['e_n_cent/ACCESS-CM2/historical/r2i1p1f1/daily/1950_2014/tasmax']
>>>>>>> 4151fd4 (initiate linux version)
    print(unit)