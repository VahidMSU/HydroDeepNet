import os 
import h5py 

path = "/data/SWATGenXApp/GenXAppData/SWATplus_by_VPUID/0000/huc12/04176500/SWAT_gwflow_MODEL/Scenarios/verification_stage_0/SWATplus_output.h5"


with h5py.File(path, 'r') as f:
    # List all groups
    soil = f['Soil/ec_30m'][:]
    print(soil.shape)
    landuse = f['Landuse/landuse_30m'][:]
    print(landuse.shape)
    perc = f['hru_wb_30m/2000/1/perc'][:]
    print(perc.shape)
    
import matplotlib.pyplot as plt

plt.imshow(soil)
plt.colorbar()
plt.savefig("./Michigan/soil.png")

plt.imshow(landuse)
plt.colorbar()
plt.savefig("./Michigan/landuse.png")