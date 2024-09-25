import os
import numpy as np
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_grid(data, title):
    plt.imshow(data, cmap='viridis')
    #plt.colorbar()
    plt.title(title)
    up_95 = np.nanpercentile(data, 95)
    low_5 = np.nanpercentile(data, 5)
    data = np.where(data > up_95, up_95, data)
    data = np.where(data < low_5, low_5, data)
    plt.colorbar()
    os.makedirs('figs_recharge_rainfall', exist_ok=True)    
    plt.savefig(f'figs_recharge_rainfall/{title}.png')
    plt.close()




def scatter_plot(x, y, title, xlabel, ylabel):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'figs_recharge_rainfall/{title}.png')
    plt.close()

def read_rainfall_recharge_landuse(path, yr=2011):
    with h5py.File(path, 'r') as f:
        # Load datasets
        rainfall = f[f'ppt_{yr}_250m'][:, :]
        landuse = f['landuse_250m'][:, :]
        recharge = f[f'recharge_{yr}_250m'][:, :]
        base_raster = f['BaseRaster_250m'][:, :]
        

        rainfall = np.where(base_raster !=1, rainfall, np.nan)
        landuse = np.where(base_raster !=1, landuse, np.nan)
        recharge = np.where(base_raster !=1, recharge, np.nan)

        rainfall = np.where(recharge ==0, np.nan, rainfall)
        landuse = np.where(recharge ==0, np.nan, landuse)
        recharge = np.where(recharge ==0, np.nan, recharge)

        rainfall = np.where(recharge ==-999, np.nan, rainfall)
        landuse = np.where(recharge ==-999, np.nan, landuse)
        recharge = np.where(recharge ==-999, np.nan, recharge)
        

        ## change recharge from m3/day to mm/year. considering the cell size of 250m*250m
        recharge = recharge * 1000 * 365 / 250 / 250
        # Further mask areas with invalid recharge
        
        scatter_plot(rainfall.flatten(), recharge.flatten(), "Rainfall vs Recharge", "Rainfall (mm/year)", "Recharge (mm/year)")
        # Plot filtered grids
        plot_grid(rainfall, "Rainfall")
        plot_grid(landuse, "Landuse")
        plot_grid(recharge, "Recharge")

        return rainfall, landuse, recharge
    
def generate_rainfall_recharge_by_landuse(rainfall, landuse, recharge, landuse_dict):
    # Find unique land-use types, excluding NaN values
    unique_landuse_types = np.unique(landuse[~np.isnan(landuse)])

    # Initialize dictionaries to hold the results
    landuse_recharge = {}
    landuse_rainfall = {}

    # Loop through each unique land-use type
    for landuse_type in unique_landuse_types:
        # Create a mask for the current land-use type
        landuse_mask = (landuse == landuse_type)
        
        # Sum recharge and rainfall for the current land-use type
        total_recharge = np.nanmean(recharge[landuse_mask])
        total_rainfall = np.nanmean(rainfall[landuse_mask])
        
        # Store the results in the dictionaries
        landuse_recharge[landuse_type] = total_recharge
        landuse_rainfall[landuse_type] = total_rainfall

    # Convert landuse types from float to string and replace them with SWAT_CODE
    landuse_recharge = {landuse_dict.get(str(int(landuse_code)), f"Unknown_{int(landuse_code)}"): recharge 
                        for landuse_code, recharge in landuse_recharge.items()}

    landuse_rainfall = {landuse_dict.get(str(int(landuse_code)), f"Unknown_{int(landuse_code)}"): rainfall 
                        for landuse_code, rainfall in landuse_rainfall.items()}
    
    return landuse_recharge, landuse_rainfall

# Load land-use lookup table
path = "/data/MyDataBase/HydroGeoDataset_ML_250.h5"
landuse_lookup = "/data/MyDataBase/SWATGenXAppData/LandUse/landuse_lookup.csv"
landuse_lookup_df = pd.read_csv(landuse_lookup)[['LANDUSE', 'SWAT_CODE']]

# Create a dictionary for easier lookup
landuse_dict = dict(zip(landuse_lookup_df['LANDUSE'].astype(str), landuse_lookup_df['SWAT_CODE']))

print(landuse_lookup_df)


rainfall, landuse, recharge = read_rainfall_recharge_landuse(path)

landuse_recharge, landuse_rainfall = generate_rainfall_recharge_by_landuse(rainfall, landuse, recharge, landuse_dict)

# Display the results
print("Annual Recharge for each Landuse type:", landuse_recharge)
print("Annual Rainfall (PPT) for each Landuse type:", landuse_rainfall)
