import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
# Load the dataset
path = "/data/MyDataBase/SWATGenXAppData/PRISM/CONUS/ppt/1992.nc"
ds = xr.open_dataset(path)
data = ds['data']
time = ds['time']

# Display dataset information
print(f"data shape: {data.shape}")  # 366, 621, 1405 (days, lat, lon)
print(f"data size: {data.size}")
print(f"data dtype: {data.dtype}")

# Replace -9999 with NaN
data = data.where(data != -9999)

# Ensure the time coordinate is properly parsed
data = data.assign_coords(time=time)

# Split data into 30-day periods
num_periods = int(np.ceil(data.shape[0] / 30))
monthly_mean_precipitation = []
period_labels = []
period_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for period_name, i in zip(period_names, range(num_periods)):
    start = i * 30
    end = min((i + 1) * 30, data.shape[0])
    period_mean = data[start:end].sum(dim='time')
    monthly_mean_precipitation.append(period_mean)
    
    
    period_labels.append(f'{period_name}')

# Plot the average 30-day period precipitation
fig, axes = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < num_periods:  # Ensure we only plot the periods we have
        pcm = ax.imshow(monthly_mean_precipitation[i], cmap='viridis')
        ax.set_title(period_labels[i])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    else:
        ax.axis('off')

plt.suptitle('Average Precipitation for 30-day Periods in 1992')
plt.show()
