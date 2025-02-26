import torch
import h5py 
import os

## environment
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'   
## visibile GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = "/data/SWATGenXApp/GenXAppData/HydroGeoDataset/SNODAS.h5"

device = torch.device('cuda:0')
print(f"device: {device}")

with h5py.File(path, 'r') as f:
    data = f['250m/2016/melt_rate']
    print(f"data has been loaded: {data}")  
    data = torch.tensor(data, device=device)
    print(data)
    import time 
    time.sleep(10)