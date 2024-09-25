import os
import pandas as pd
import geopandas as gpd
import numpy as np

# Load the GeoJSON file
shape = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples/PFAS_Surface_Water_Sampling.geojson"
pfas_df = gpd.read_file(shape)
print(f"CRS: {pfas_df.crs}")
import time

time.sleep(500)
# Print some characteristics including the number of rows, columns, and the first 5 rows
print(f"Number of rows: {pfas_df.shape[0]}")
print(f"Number of columns: {pfas_df.shape[1]}")
print(f"Columns: {list(pfas_df.columns)}")

# Extract PFAS concentration columns
concentration_columns = [col for col in pfas_df.columns if 'CAS' in col and not any(x in col for x in ['Flag', 'Mdl', 'Rl'])]

# Function to compute statistics
def compute_statistics(pfas_df, concentration_columns):
	stats = {col: {'min': [], 'max': [], 'mean': [], 'std': []} for col in concentration_columns}
	coords = [(geom.x, geom.y) for geom in pfas_df.geometry]

	for conc_col in concentration_columns:
		values = pfas_df[conc_col].values
		for x, y in coords:
			mask = (pfas_df.geometry.x == x) & (pfas_df.geometry.y == y)
			if not mask.any():
				continue
			cell_values = values[mask]
			if np.isnan(cell_values).all():
				continue
			min_value = np.nanmin(cell_values)
			max_value = np.nanmax(cell_values)
			mean_value = np.nanmean(cell_values)
			std_value = np.nanstd(cell_values)
			stats[conc_col]['min'].append((x, y, min_value))
			stats[conc_col]['max'].append((x, y, max_value))
			stats[conc_col]['mean'].append((x, y, mean_value))
			stats[conc_col]['std'].append((x, y, std_value))

	return stats

# Compute statistics for each concentration column
stats = compute_statistics(pfas_df, concentration_columns)

# Function to create GeoDataFrame from statistics
def create_gdf_from_stats(stats, stat_type):
	data = []
	for conc_col in stats:
		for x, y, value in stats[conc_col][stat_type]:
			data.append({'geometry': gpd.points_from_xy([x], [y])[0], 'concentration': value, 'compound': conc_col})
	return gpd.GeoDataFrame(data, crs=pfas_df.crs)




# Create GeoDataFrames for min, max, and mean
gdf_min = create_gdf_from_stats(stats, 'min')
gdf_max = create_gdf_from_stats(stats, 'max')
gdf_mean = create_gdf_from_stats(stats, 'mean')
gdf_std = create_gdf_from_stats(stats, 'std')
# Save GeoDataFrames as shapefiles
output_dir = "/data/MyDataBase/SWATGenXAppData/PFAS_surface_water_samples"
gdf_min.to_pickle(os.path.join(output_dir, 'PFAS_Min.pkl'))
gdf_max.to_pickle(os.path.join(output_dir, 'PFAS_Max.pkl'))
gdf_mean.to_pickle(os.path.join(output_dir, 'PFAS_Mean.pkl'))
gdf_std.to_pickle(os.path.join(output_dir, 'PFAS_Std.pkl'))
print("Shapefiles created successfully.")
