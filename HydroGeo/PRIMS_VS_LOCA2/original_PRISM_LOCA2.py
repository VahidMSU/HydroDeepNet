import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

PRISM = '/data/PRISM/Michigan_250m_1990_2022.h5'
LOCA2 = '/data/MyDataBase/LOCA2_MLP.h5'

with h5py.File(PRISM, 'r') as f:
    print(f.keys())
    prism_data = f['ppt/1990'][:]
    ## show one day
    plt.imshow(prism_data[0])
    plt.colorbar()
    os.makedirs('figs', exist_ok=True)  
    plt.savefig('figs/prism.png')
    plt.close()

## now LOCA2
with h5py.File(LOCA2, 'r') as f:
    print(f['e_n_cent/ACCESS-CM2/historical/r1i1p1f1/daily/1950_2014/pr'].shape)
    loca2_data = f['e_n_cent/ACCESS-CM2/historical/r1i1p1f1/daily/1950_2014/pr'][:]
    ## flip the axis
    loca2_data = np.flip(loca2_data, axis=1)
    ## show one day
    plt.imshow(loca2_data[0])
    plt.colorbar()
    plt.savefig('figs/loca2.png')
    plt.close()