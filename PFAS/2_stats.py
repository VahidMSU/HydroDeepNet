import pandas as pd
import os
path = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples"
gdf_min = pd.read_pickle(os.path.join(path, 'PFAS_Min.pkl'))
gdf_max = pd.read_pickle(os.path.join(path, 'PFAS_Max.pkl'))
gdf_mean = pd.read_pickle(os.path.join(path, 'PFAS_Mean.pkl'))
gdf_std = pd.read_pickle(os.path.join(path, 'PFAS_Std.pkl'))
print(f"list of unique compounds: {gdf_min['compound'].unique()}")
## print some stats with respect to each column of the data
### min max mean for each compound
print(f"Min, Max, and Mean values for each compound:")
for compound in gdf_min['compound'].unique():
	min_value = gdf_min[gdf_min['compound'] == compound]['concentration'].values
	max_value = gdf_max[gdf_max['compound'] == compound]['concentration'].values
	mean_value = gdf_mean[gdf_mean['compound'] == compound]['concentration'].values
	std_value = gdf_std[gdf_std['compound'] == compound]['concentration'].values
	print(f"Compound: {compound}")
	print(f"Min: {min_value.min()}, Max: {max_value.max()}, Mean: {mean_value.mean()}", f"Std: {std_value.std()}")

	with open('stats.txt', 'a') as f:
		f.write(f"Compound: {compound}\n")
		f.write(f"Min: {min_value.min()}, Max: {max_value.max()}, Mean: {mean_value.mean()}, Std: {std_value.std()}\n")