import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os


path = "/data/MyDataBase/SWATGenXAppData/SWAT_input/huc8/4100013/SWAT_plus_Subbasin/SWAT_plus_Subbasin.shp"
sites =  "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SITE_Features.pkl"
pfas_gw = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_GW_Features.pkl"
sites = gpd.GeoDataFrame(pd.read_pickle(sites)).to_crs('EPSG:4326')
pfas_gw = gpd.GeoDataFrame(pd.read_pickle(pfas_gw)).to_crs('EPSG:4326')

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

path = "/data/MyDataBase/SWATGenXAppData/SWAT_input/huc8/4100013/SWAT_plus_Subbasin/SWAT_plus_Subbasin.shp"
sites_path =  "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_SITE_Features.pkl"
pfas_gw_path = "/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/input_data/Huron_PFAS_GW_Features.pkl"
sites = gpd.GeoDataFrame(pd.read_pickle(sites_path)).to_crs('EPSG:4326')
pfas_gw = gpd.GeoDataFrame(pd.read_pickle(pfas_gw_path)).to_crs('EPSG:4326')

def calculate_sum_PFAS_concentration(pfas_gw):
    ### first fill nan values with zeros
    pfas_gw.fillna(0, inplace=True)
    ## then groupby WSSN and sum the values
    pfas_gw_sum = pfas_gw[['WSSN', 'PFHxAResult', 'PFBSResult', "PFHpAResult", "PFOAResult", "PFHxSResult", "PFOSResult", "NEtFOSAAResult", "NMeFOSAAResult"]].groupby('WSSN').sum().reset_index()
    ### now calculate the sum of all PFAS, except WSSN
    pfas_gw_sum['sum_PFASResult'] = pfas_gw_sum.iloc[:, 1:].sum(axis=1)
    pfas_gw_sum = pfas_gw_sum[['WSSN', 'sum_PFASResult']]
    pfas_gw_sum = pfas_gw_sum.merge(pfas_gw[['WSSN', 'geometry']], on='WSSN')

    ### make sure its a geodataframe
    pfas_gw_sum = gpd.GeoDataFrame(pfas_gw_sum, crs='EPSG:4326')

    assert "geometry" in pfas_gw_sum.columns, "The geometry column is missing"

    plt.figure(figsize=(10, 10))
    bounds = gpd.read_file(path).dissolve().to_crs('EPSG:26990')
    bounds = bounds.simplify(100).to_crs('EPSG:4326')
    bounds.plot(color='lightgrey', edgecolor='black', linewidth=0.25)
    sites.plot(ax=plt.gca(), color='black', markersize=10, marker='^')
    pfas_gw_sum[pfas_gw_sum['sum_PFASResult'] == 0].plot(ax=plt.gca(), color='green', markersize=10, marker='o', edgecolor='black', linewidth=0.5)
    pfas_gw_sum[pfas_gw_sum['sum_PFASResult'] > 0].plot(ax=plt.gca(), color='orange', markersize=15, marker='o', edgecolor='red', linewidth=0.5)
    plt.legend(["PFAS Sites", "Not Detected", "Detected"])
    plt.title("Sum of PFAS")
    plt.xticks(np.round(np.linspace(bounds.bounds.minx.min(), bounds.bounds.maxx.max(), 4), 1))
    plt.yticks(np.round(np.linspace(bounds.bounds.miny.min(), bounds.bounds.maxy.max(), 4), 1))
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_maps", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_maps/sum_PFAS.png", dpi=300)
    plt.close()

    return pfas_gw_sum

calculate_sum_PFAS_concentration(pfas_gw)



target_pfas_species = ['PFHxA', 'PFBS', "PFHpA", "PFOA", "PFHxS", "PFOS", "NEtFOSAA", "NMeFOSAA"]
bounds = gpd.read_file(path).dissolve().to_crs('EPSG:26990')
bounds = bounds.simplify(100).to_crs('EPSG:4326')

for species in target_pfas_species:
    ### simplify the boundary
    plt.figure(figsize = (10, 10))
    bounds.plot(color = 'lightgrey', edgecolor = 'black', linewidth = 0.25)
    ### use triangle to plot the sites
    sites.plot(ax = plt.gca(), color = 'black', markersize = 10, marker = '^')
    ##### two plots:1- possitive values, 2- zero values
    pfas_gw[pfas_gw[species+"Result"] == 0].plot(ax = plt.gca(), color = 'green', markersize = 10, marker = 'o', edgecolor = 'black', linewidth = 0.5)
    pfas_gw[pfas_gw[species+"Result"] > 0].plot(ax = plt.gca(), color = 'orange', markersize = 15, marker = 'o', edgecolor = 'red', linewidth = 0.5)
    plt.legend(["PFAS Sites", "Not Detected", "Detected"])
    plt.title(species)
    ### only three tickes with 1 decimal point
    plt.xticks(np.round(np.linspace(bounds.bounds.minx.min(), bounds.bounds.maxx.max(), 4), 1))
    plt.yticks(np.round(np.linspace(bounds.bounds.miny.min(), bounds.bounds.maxy.max(), 4), 1))
    plt.grid(axis = 'both', linestyle = '--', alpha = 0.5)
    os.makedirs("/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_maps", exist_ok = True)
    plt.tight_layout()
    plt.savefig(f"/data/MyDataBase/SWATGenXAppData/codes/PFAS_SW_GW_HuronRiver/spatial_maps/{species}.png", dpi=300)
    plt.close()