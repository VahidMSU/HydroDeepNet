import pandas as pd 
path = "/data/MyDataBase/SWATGenXAppData/GW-Machine-Learning/grid_points_well_obs_with_geometry.pk1"
df = pd.read_pickle(path)   
print(df.columns)

