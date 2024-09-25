import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt

def count_observations(gws_path):
    """Count the number of observations for each groundwater
    station and add the result to the shapefile."""
    dgw = gpd.read_file(stations_locations_df)
    observations_no = []  # number of observations

    for station in dgw.site_no:
        path = os.path.join(gws_path, f'{station}.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            number_of_observations = len(data[' head_m'].drop_duplicates())
            observations_no.append([number_of_observations, station])

    result = pd.DataFrame(observations_no, columns=['num_of_ob', 'site_no'])
    dgw = dgw.merge(result, on='site_no')
    dgw.to_file(output, driver='ESRI Shapefile')
    return dgw

gws_path = r"/data/MyDataBase/SWATGenXAppData/groundwater_daily_stations"
stations_locations_df = os.path.join(gws_path, "statons_location\statons_location.shp")
output = os.path.join(gws_path, "statons_location\statons_location_with_ob_num.shp")

if os.path.exists(output):
    print(f"{output} already exists")
else:
    dgw = count_observations(gws_path)

gw = gpd.read_file(output).to_crs("EPSG:4326")

# Discretize 'num_of_ob' into categories
# Adjust the bins according to your specific needs
bins = [0, 100, 500, 1000, 15000]  # Example bins
labels = ['0-100', '100-500', '501-1000', '1001-15000']  # Corresponding labels for bins
gw['num_of_ob_cat'] = pd.cut(gw['num_of_ob'], bins=bins, labels=labels, include_lowest=True)

# Now plot using these categories
fig, ax = plt.subplots(figsize=(10, 10))
gw.plot(column='num_of_ob_cat', legend=True, cmap='viridis', ax=ax, edgecolor='black',
        legend_kwds={'title': "Number of Observations", 'loc': 'upper left'})

boundary = gpd.read_file(r"D:\MyDataBase\Boundary\Michigan_LP_boundary_reprojected.shp").to_crs("EPSG:4326")
boundary.plot(ax=ax, color='none', edgecolor='black')

plt.title("USGS Groundwater Stations in Michigan\nNumber of Observations per Station")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
os.makedirs(r"D:\MyDataBase\Documentations\Groundwater", exist_ok=True)
plt.savefig(r'D:\MyDataBase\Documentations\Groundwater\Groundwater_stations.png', dpi=300)
plt.show()
