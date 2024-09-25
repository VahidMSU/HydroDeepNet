import os
import geopandas as gpd
import pandas as pd
NAMES = os.listdir("/data/MyDataBase/SWATplus_by_VPUID/0000/huc12/")
model_bounds = pd.read_pickle('model_bounds/huc12_model_bounds.pkl')

print('model bounds are fetched', model_bounds.head())
print(f'model bounds columns are {model_bounds.columns}')