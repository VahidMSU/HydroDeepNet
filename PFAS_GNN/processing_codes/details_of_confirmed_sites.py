import pandas as pd
import geopandas as gpd


confirmed_pfas_sites = "/data/MyDataBase/HuronRiverPFAS/Huron_confirmed_PFAS_SITE_Features.pkl"

df = pd.read_pickle(confirmed_pfas_sites)
print(f"Number of Confirmed PFAS sites: {df.shape[0]}")
print(f"Unique Type in Confirmed PFAS sites: {df.Type.unique()}")
print(f"Facility Type in Confirmed PFAS sites: {df.Facility.unique()}")

### number of Unique for each Type
for t in df.Type.unique():
    print(f"Number of {t} sites: {df[df.Type == t].shape[0]}")
    


