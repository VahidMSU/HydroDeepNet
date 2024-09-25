import pandas as pd
import matplotlib.pyplot as plt

biosolids_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_Biosolid_sites_with_features.pkl"
confirmed_pfas_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_confirmed_PFAS_SITE_Features.pkl"
unconfirmed_pfas_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_PFAS_SITE_Features.pkl"
bound = "/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl"

biosolids_sites = pd.read_pickle(biosolids_sites).to_crs(epsg=4326)
print(f"Number of Biosolid sites: {len(biosolids_sites)}")
import time 
time.sleep(50)
confirmed_pfas_sites = pd.read_pickle(confirmed_pfas_sites).to_crs(epsg=4326)

unconfirmed_pfas_sites = pd.read_pickle(unconfirmed_pfas_sites).to_crs(epsg=4326)
bound = pd.read_pickle(bound).to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(7, 7))
biosolids_sites.plot(ax=ax, color='green', markersize=14)
unconfirmed_pfas_sites.plot(ax=ax, color='blue', markersize=20)
confirmed_pfas_sites.plot(ax=ax, color='red', markersize=18)
bound.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
plt.legend(["Biosolid leaching", "Suspected PFAS Sites", "Confirmed PFAS Sites"], loc = 'upper left')
plt.title("Confirmed and suspected PFAS sources in Huron River Basin")
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
## remove x and y axis
plt.xticks([])
plt.yticks([])

plt.grid(alpha=0.5, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("figs/Michigan_confirmed_PFAS_Sites.png", dpi=600)
plt.close()
