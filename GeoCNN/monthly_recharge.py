import h5py 
import numpy as np
import matplotlib.pyplot as plt

path = "/data/MyDataBase/SNODAS.h5"
parameters = ['melt_rate', 'snow_accumulation', 'snow_layer_thickness', 'snow_water_equivalent', 'snowpack_sublimation_rate']

def plot_annual(annual_melt_rate, no_value=0):
    annual_melt_rate = np.where(annual_melt_rate == no_value, np.nan, annual_melt_rate)
    plt.imshow(annual_melt_rate, cmap='viridis')
    plt.colorbar()
    plt.savefig('figs/annual_melt_rate.png')

with h5py.File(path, 'r') as f:
    print(f['250m/2004/melt_rate'])
    annual_melt_rate = f['250m/2004/melt_rate'][:]
    print(annual_melt_rate)
    ## replace 55537 with nan
    annual_melt_rate[annual_melt_rate == 55537] = 0
    # calculate montly average melt rate
    for i in range(1, 366):
        if i % 30 == 0:
            print(i)
            monthly_melt_rate = np.mean(annual_melt_rate[i-30:i, :, :], axis=0)
            #monthly_melt_rate = np.where(monthly_melt_rate == 0, np.nan, monthly_melt_rate)
            plt.imshow(monthly_melt_rate, cmap='viridis')
            plt.colorbar()
            plt.savefig(f'figs/monthly_melt_rate_{i}.png')
            #print(monthly_melt_rate)
            plt.close()
        



