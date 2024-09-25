import geopandas as gpd
from libs.load_data import load_pfas_sites
from libs.utils import setup_logging
import matplotlib.pyplot as plt
import os
import pandas as pd
data_dir =[ '/data/MyDataBase/HuronRiverPFAS/']
logger = setup_logging()
pfas_sites = load_pfas_sites(data_dir = "/data/MyDataBase/HuronRiverPFAS/", logger = logger)
pfas_sites =pfas_sites.to_crs("EPSG:4326")
bounds = gpd.read_file("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.geojson")
bounds = bounds.to_crs("EPSG:4326")
## red for inv_status = 1, green for inv_status = 0
pfas_sites_inv_status_1 = pfas_sites[pfas_sites['inv_status'] == 1]
pfas_sites_inv_status_0 = pfas_sites[pfas_sites['inv_status'] == 0]
fig, ax = plt.subplots()
plt.grid(alpha=0.5, linestyle='--')
bounds.plot(ax=ax, facecolor='none', edgecolor='black')

pfas_sites_inv_status_0.plot(ax=ax, color='blue', markersize=5)
pfas_sites_inv_status_1.plot(ax=ax, color='red', markersize=8)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
## remove x and y axis
plt.xticks([])
plt.yticks([])
plt.title('PFAS sites')
plt.legend(['Confirmed PFAS site', 'Suspected PFAS site'])
plt.tight_layout()    
plt.savefig('figs/confirmed_suspected_pfas.png', dpi=300)