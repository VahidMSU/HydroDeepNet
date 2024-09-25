import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd
path = "/data/MyDataBase/CIWRE-BAE/lakes/"
lakes_name = os.listdir(path)
##only those ending with zip
lakes_name = [lake for lake in lakes_name if lake.endswith(".zip")]
### drop zip extension
lakes_name = [lake.split(".")[0] for lake in lakes_name]
print(lakes_name)
lakes = []
for lake in lakes_name:
    lake_data = gpd.read_file(os.path.join(path, lake)).to_crs('EPSG:4326')
    ## clip by max and min lat and lon
    max_lat = -82.0
    max_lon = 45.0
    min_lat = -90.0
    min_lon = 40.0
    ## now clip the data
    lake_data = lake_data.cx[min_lat:max_lat, min_lon:max_lon]
    ## dissolve the data
    lake_data = lake_data.dissolve()
    lakes.append(lake_data)


lakes = gpd.GeoDataFrame(pd.concat(lakes, ignore_index=True))
## save the model bounds
os.makedirs("model_bounds", exist_ok=True)
lakes.to_pickle(f"model_bounds/lakes_model_bounds.pkl")
#plot
fig, ax = plt.subplots(figsize=(15, 10))
lakes.boundary.plot(ax=ax, color='black', linewidth=1)
lakes.plot(ax=ax, alpha=1,  edgecolor='black', linewidth=1, facecolor='skyblue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'# {len(lakes)} model bounds')
plt.savefig(f'model_bounds/lakes_model_bounds.png', dpi=300)
